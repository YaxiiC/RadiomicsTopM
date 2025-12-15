import torch
import numpy as np
import math
import logging
import warnings

import SimpleITK as sitk


from torchradiomics import (
    TorchRadiomicsGLCM,
    TorchRadiomicsGLDM,
    TorchRadiomicsGLRLM,
    TorchRadiomicsGLSZM,
    TorchRadiomicsNGTDM,
    TorchRadiomicsFirstOrder,
    inject_torch_radiomics
)
logging.getLogger("torchradiomics").setLevel(logging.ERROR)


def first_order_and_shape_features(x, voxelArrayShift=0, pixelSpacing=[1.0, 1.0, 1.0], binWidth=None):
    # Apply voxel array shift to avoid negative values
    x = x + voxelArrayShift
    Np = x.numel()
    X = x.view(-1)

    device = X.device 

    # histogram
    if binWidth:
        min_val, max_val = X.min(), X.max()
        bins = int(((max_val - min_val) / binWidth).item()) + 1
        hist = torch.histc(X, bins=bins, min=min_val.item(), max=max_val.item())
    else:
        Ng = 256
        min_val, max_val = X.min(), X.max()
        hist = torch.histc(X, bins=Ng, min=min_val.item(), max=max_val.item())

    p_i = hist / Np
    p_i = p_i[p_i > 0]

    # voxel volume
    voxel_volume = torch.tensor(pixelSpacing, device=device).prod()

    # first-order features
    energy  = torch.sum(X ** 2)
    total_energy = voxel_volume * energy
    eps     = torch.tensor(2.2e-16, device=device)
    entropy = -torch.sum(p_i * torch.log2(p_i + eps))
    minimum = X.min()
    perc10  = torch.quantile(X, 0.1)
    perc90  = torch.quantile(X, 0.9)
    maximum = X.max()
    mean    = X.mean()
    median  = X.median()
    iqr     = torch.quantile(X, 0.75) - torch.quantile(X, 0.25)
    range_  = maximum - minimum
    mad     = torch.mean(torch.abs(X - mean))

    # robust MAD
    X_10_90 = X[(X >= perc10) & (X <= perc90)]
    mean_10_90 = torch.mean(X_10_90)
    rmad    = torch.mean(torch.abs(X_10_90 - mean_10_90))

    rms     = torch.sqrt(torch.mean(X ** 2))
    std     = X.std()
    if std.item() != 0:
        skew   = torch.mean(((X - mean) / std) ** 3)
        kurt   = torch.mean(((X - mean) / std) ** 4)
    else:
        skew, kurt = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
    var     = X.var()
    uniform = torch.sum(p_i ** 2)

    # shape
    volume, surface_area = _calculate_geometric_features(x, pixelSpacing)
    eigvals, diameters, max2d = _calculate_eigenvalues_and_diameters(x, pixelSpacing)

    # assemble
    features = {
        "Energy": energy,
        "TotalEnergy": total_energy,
        "Entropy": entropy,
        "Minimum": minimum,
        "10Percentile": perc10,
        "90Percentile": perc90,
        "Maximum": maximum,
        "Mean": mean,
        "Median": median,
        "InterquartileRange": iqr,
        "Range": range_,
        "MeanAbsoluteDeviation": mad,
        "RobustMeanAbsoluteDeviation": rmad,
        "RootMeanSquared": rms,
        "StandardDeviation": std,
        "Skewness": skew,
        "Kurtosis": kurt,
        "Variance": var,
        "Uniformity": uniform,
        "MeshVolume": volume,
        "VoxelVolume": volume * torch.tensor(pixelSpacing, device=device).prod(),
        "SurfaceArea": surface_area,
        "SurfaceVolumeRatio": surface_area / volume if volume > 0 else 0,
        "Sphericity": _calculate_sphericity(volume, surface_area),
        "Compactness1": _calculate_compactness1(volume, surface_area),
        "Compactness2": _calculate_compactness2(volume, surface_area),
        "Flatness": _calculate_flatness(eigvals),
        "Elongation": _calculate_elongation(eigvals),
        "LeastAxisLength": diameters[0],
        "MinorAxisLength": diameters[1],
        "MajorAxisLength": diameters[2],
        "Maximum2DDiameterColumn": max2d[0],
        "Maximum2DDiameterRow":    max2d[1],
        "Maximum2DDiameterSlice":  max2d[2],
        "Maximum3DDiameter":       diameters[2],
    }

    # validate
    valids = 0
    invalids = []
    for k,v in features.items():
        if isinstance(v, torch.Tensor):
            if torch.isnan(v).any() or torch.isinf(v).any():
                invalids.append(k)
            else:
                valids += 1
        else:
            if math.isnan(v) or math.isinf(v):
                invalids.append(k)
            else:
                valids += 1

    #print(f"[INFO] First-order & shape valid: {valids}, invalid: {invalids}")
    return features, list(features.keys())


def _calculate_geometric_features(image, voxel_spacing):
    if image.dim() == 3:
        image = image.unsqueeze(0).unsqueeze(0)
    voxel_vol = torch.tensor(voxel_spacing).prod()
    vol = (image > 0).sum().float() * voxel_vol

    boundary = torch.logical_xor(
        image > 0,
        torch.nn.functional.max_pool3d(image.float(), 3, stride=1, padding=1) > 0
    )
    vsurf = 2 * (
        voxel_spacing[1]*voxel_spacing[2] +
        voxel_spacing[0]*voxel_spacing[2] +
        voxel_spacing[0]*voxel_spacing[1]
    )
    surf = boundary.sum().float() * vsurf
    return vol, surf


def _calculate_eigenvalues_and_diameters(image, voxel_spacing):
    coords = torch.nonzero(image > 0).float()
    if coords.numel() == 0:
        zero = torch.zeros(3, device=image.device)
        return zero, zero, zero

    coords = coords[:, :3] * torch.tensor(voxel_spacing, device=image.device)
    centered = coords - coords.mean(dim=0, keepdim=True)
    cov = centered.T @ centered / (coords.size(0) - 1)
    eigvals = torch.linalg.eigvalsh(cov)
    eigvals, _ = torch.sort(eigvals)
    diameters = 4 * torch.sqrt(eigvals)

    # 2D diameters
    col = coords[:,1].max() - coords[:,1].min()
    row = coords[:,2].max() - coords[:,2].min()
    slc = coords[:,0].max() - coords[:,0].min()
    return eigvals, diameters, torch.tensor([col,row,slc], device=image.device)


def _calculate_sphericity(vol, surf):
    return (36*torch.pi*vol**2)**(1/3) / surf if surf>0 else 0.0


def _calculate_compactness1(vol, surf):
    return vol / torch.sqrt(torch.pi * surf**3) if surf>0 else 0.0


def _calculate_compactness2(vol, surf):
    return (36*torch.pi*vol**2) / (surf**3) if surf>0 else 0.0


def _calculate_flatness(eigvals):
    return float(torch.sqrt(eigvals[0] / (eigvals[-1] + 1e-9))) if eigvals[-1]>0 else 0.0


def _calculate_elongation(eigvals):
    return float(torch.sqrt(eigvals[1] / (eigvals[-1] + 1e-9))) if eigvals[-1]>0 else 0.0


def extract_all_radiomics(x, voxelArrayShift=0, pixelSpacing=[1.0,1.0,1.0], binWidth=None):
    # 1) first-order + shape
    fo_dict, fo_names = first_order_and_shape_features(x, voxelArrayShift, pixelSpacing, binWidth)

    # 2) PyTorch → NumPy → SITK image + mask
    img_np  = x.to(dtype=torch.float64, device=x.device).cpu().numpy()
    mask_np = (x > 0).to(dtype=torch.uint8, device=x.device).cpu().numpy()
    sitk_img  = sitk.GetImageFromArray(img_np)
    sitk_mask = sitk.GetImageFromArray(mask_np)
    # SITK spacing expects (x,y,z)
    sitk_img.SetSpacing((pixelSpacing[2], pixelSpacing[1], pixelSpacing[0]))
    sitk_mask.SetSpacing((pixelSpacing[2],pixelSpacing[1],pixelSpacing[0]))

    # 3) Inject defaults & build extractors
    inject_torch_radiomics()
    base_kwargs = dict(
        voxelBased=False,
        padDistance=1,
        kernelRadius=1,
        maskedKernel=False,
        voxelBatch=512,
        dtype=torch.float64,
        device= x.device
    )
    extractors = [
        TorchRadiomicsGLCM( sitk_img, sitk_mask, **base_kwargs),
        TorchRadiomicsGLDM( sitk_img, sitk_mask, **base_kwargs),
        TorchRadiomicsGLRLM(sitk_img, sitk_mask, **base_kwargs),
        TorchRadiomicsGLSZM(sitk_img, sitk_mask, **base_kwargs),
        TorchRadiomicsNGTDM(sitk_img,sitk_mask, **base_kwargs),
    ]

    # 4) execute & collect
    matrix_dict  = {}
    matrix_names = []
    for ext in extractors:
        feats = ext.execute()
        for k, v in feats.items():
            # skip any feature maps (SimpleITK Image) and non-numerical entries
            if isinstance(v, sitk.Image):
                continue
            # convert scalar to tensor
            matrix_dict[k] = torch.as_tensor(v, device=x.device) \
                             if not isinstance(v, torch.Tensor) else v
            matrix_names.append(k)

    # merge
    all_dict  = {**fo_dict, **matrix_dict}
    all_names = fo_names + matrix_names
    return all_dict, all_names


if __name__ == "__main__":
    # minimal smoke test
    test_img = torch.rand(20,20,20) * 100
    test_img[test_img<20] = 0

    feats, names = extract_all_radiomics(
        test_img,
        voxelArrayShift=0,
        pixelSpacing=[1.0,1.0,1.0],
        binWidth=5.0
    )

    print(f"Extracted {len(names)} features:")
    for nm in names[:100]:
        val = feats[nm]
        if isinstance(val, torch.Tensor): val = val.item()
        print(f"  {nm:25s} = {val:.4f}")

    # check validity
    bad = [n for n,v in feats.items()
           if (isinstance(v,torch.Tensor) and (torch.isnan(v)|torch.isinf(v)).any())]
    print("[OK]" if not bad else f"[ERROR] Invalid: {bad}")