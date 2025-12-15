import torch
import numpy as np
import torch.nn.functional as F
import math

def first_order_and_shape_features(x, voxelArrayShift=0, pixelSpacing=[1.0, 1.0, 1.0], binWidth=None):
    # Apply voxel array shift to avoid negative values
    x = x + voxelArrayShift
    Np = x.numel()
    X = x.view(-1)

    device = X.device 

    if binWidth:
        min_val = X.min()
        max_val = X.max()
        bins = int(((max_val - min_val) / binWidth).item()) + 1
        hist = torch.histc(X, bins=bins, min=min_val.item(), max=max_val.item())
    else:
        Ng = 256
        min_val = X.min()
        max_val = X.max()
        hist = torch.histc(X, bins=Ng, min=min_val.item(), max=max_val.item())

    p_i = hist / Np
    p_i = p_i[p_i > 0]

    # Calculate voxel volume for total energy (assume isotropic voxels if spacing is not specified)
    voxel_volume = torch.tensor(pixelSpacing, device=device).prod()

    # Feature calculations
    energy = torch.sum(X ** 2)
    total_energy = voxel_volume * energy

    eps = torch.tensor(2.2e-16, device=device)
    entropy = -torch.sum(p_i * torch.log2(p_i + eps))

    minimum = X.min()
    percentile_10 = torch.quantile(X, 0.1)
    percentile_90 = torch.quantile(X, 0.9)
    maximum = X.max()
    mean = X.mean()
    median = X.median()
    interquartile_range = torch.quantile(X, 0.75) - torch.quantile(X, 0.25)
    range_ = maximum - minimum
    mean_absolute_deviation = torch.mean(torch.abs(X - mean))
    
    # Calculate rMAD (Robust Mean Absolute Deviation)
    X_10_90 = X[(X >= percentile_10) & (X <= percentile_90)]
    mean_10_90 = torch.mean(X_10_90)
    robust_mean_absolute_deviation = torch.mean(torch.abs(X_10_90 - mean_10_90))

    # RMS calculation with voxel shift
    root_mean_squared = torch.sqrt(torch.mean(X ** 2))

    # Standard deviation, skewness, kurtosis, variance, and uniformity
    standard_deviation = X.std()
    if standard_deviation.item() != 0:
        skewness = torch.mean(((X - mean) / standard_deviation) ** 3)
        kurtosis = torch.mean(((X - mean) / standard_deviation) ** 4)
    else:
        skewness = torch.tensor(0.0, device=device)
        kurtosis = torch.tensor(0.0, device=device)
    variance = X.var()
    uniformity = torch.sum(p_i ** 2)

    # Shape-related features
    volume, surface_area = _calculate_geometric_features(x, pixelSpacing)
    eigenvalues, diameters, max_2d_diameters = _calculate_eigenvalues_and_diameters(x, pixelSpacing)

    feature_names = [
        "Energy", "TotalEnergy", "Entropy", "Minimum", "10Percentile", "90Percentile",
        "Maximum", "Mean", "Median", "InterquartileRange", "Range", "MeanAbsoluteDeviation",
        "RobustMeanAbsoluteDeviation", "RootMeanSquared", "StandardDeviation", "Skewness",
        "Kurtosis", "Variance", "Uniformity", "MeshVolume", "VoxelVolume", "SurfaceArea",
        "SurfaceVolumeRatio", "Sphericity", "Compactness1", "Compactness2", "Flatness",
        "Elongation", "LeastAxisLength", "MinorAxisLength", "MajorAxisLength",
        "Maximum2DDiameterColumn", "Maximum2DDiameterRow", "Maximum2DDiameterSlice",
        "Maximum3DDiameter", "loc_i", "loc_j", "loc_k", "loc_i_recon", "loc_j_recon", "loc_k_recon"
    ]

    # Combine all features into a single dictionary
    features_dict = {
        # First-order statistics
        "Energy": energy,
        "TotalEnergy": total_energy,
        "Entropy": entropy,
        "Minimum": minimum,
        "10Percentile": percentile_10,
        "90Percentile": percentile_90,
        "Maximum": maximum,
        "Mean": mean,
        "Median": median,
        "InterquartileRange": interquartile_range,
        "Range": range_,
        "MeanAbsoluteDeviation": mean_absolute_deviation,
        "RobustMeanAbsoluteDeviation": robust_mean_absolute_deviation,
        "RootMeanSquared": root_mean_squared,
        "StandardDeviation": standard_deviation,
        "Skewness": skewness,
        "Kurtosis": kurtosis,
        "Variance": variance,
        "Uniformity": uniformity,
        # Shape features
        "MeshVolume": volume,
        "VoxelVolume": volume * torch.prod(torch.tensor(pixelSpacing, dtype=torch.float32)),
        "SurfaceArea": surface_area,
        "SurfaceVolumeRatio": surface_area / volume if volume > 0 else 0,
        "Sphericity": _calculate_sphericity(volume, surface_area),
        "Compactness1": _calculate_compactness1(volume, surface_area),
        "Compactness2": _calculate_compactness2(volume, surface_area),
        "Flatness": _calculate_flatness(eigenvalues),
        "Elongation": _calculate_elongation(eigenvalues),
        "LeastAxisLength": diameters[0],
        "MinorAxisLength": diameters[1],
        "MajorAxisLength": diameters[2],
        "Maximum2DDiameterColumn": max_2d_diameters[0],
        "Maximum2DDiameterRow": max_2d_diameters[1],
        "Maximum2DDiameterSlice": max_2d_diameters[2],
        "Maximum3DDiameter": diameters[2]
    }
    valid_feature_count = 0
    invalid_features = []

    for key, value in features_dict.items():
        if isinstance(value, torch.Tensor):
            if torch.isnan(value).any() or torch.isinf(value).any():
                invalid_features.append(key)
                print(f"[ERROR] Feature '{key}' has invalid value: {value.item()}")
            else:
                valid_feature_count += 1
        else:
            if math.isnan(value) or math.isinf(value):
                print(f"[ERROR] Feature '{key}' has invalid value: {value}")
                invalid_features.append(key)
            else:
                valid_feature_count += 1
    print(f"[INFO] Number of valid features extracted: {valid_feature_count}")
    if invalid_features:
        print(f"[WARNING] The following features have invalid values: {invalid_features}")
    print(f"[INFO] Number of valid features extracted: {valid_feature_count}")

    return features_dict, feature_names



def _calculate_geometric_features(image, voxel_spacing):
    """
    Calculate basic geometric features such as volume and surface area.
    
    Args:
    image (torch.Tensor): A 3D or 4D tensor of shape (Depth, Height, Width).
    voxel_spacing (list): The spacing of the voxels in each dimension (Depth, Height, Width).
    
    Returns:
    tuple: (volume, surface_area)
    """
    if image.dim() == 3:
        image = image.unsqueeze(0).unsqueeze(0) 
    voxel_volume = torch.prod(torch.tensor(voxel_spacing, dtype=torch.float32))
    volume = torch.sum(image > 0).float() * voxel_volume

    # Approximate surface area using boundary voxel count and voxel surface area scaling
    boundary_voxels = torch.logical_xor(image > 0, torch.nn.functional.max_pool3d(image.float(), 3, stride=1, padding=1) > 0)
    voxel_surface_area = 2 * (voxel_spacing[1] * voxel_spacing[2] + voxel_spacing[0] * voxel_spacing[2] + voxel_spacing[0] * voxel_spacing[1])
    surface_area = boundary_voxels.sum().float() * voxel_surface_area # Adjustment factor for partial voxel contributions

    return volume, surface_area

def _calculate_eigenvalues_and_diameters(image, voxel_spacing):
    """
    Calculate eigenvalues using Principal Component Analysis (PCA) for shape description.

    Args:
    image (torch.Tensor): A 3D tensor of shape (Depth, Height, Width).
    voxel_spacing (list): The spacing of the voxels in each dimension (Depth, Height, Width).

    Returns:
    torch.Tensor: A tensor of sorted eigenvalues (smallest to largest).
    """
    coordinates = torch.nonzero(image > 0).float()
    #print("Coordinates shape:", coordinates.shape)
    if coordinates.numel() == 0:
        return torch.zeros(3, device=image.device), torch.zeros(3, device=image.device), torch.zeros(3, device=image.device)

    # Scale coordinates by voxel spacing
    coordinates = coordinates[:, :3] 
    voxel_spacing_tensor = torch.tensor(voxel_spacing, dtype=torch.float32, device=image.device)
    coordinates *= voxel_spacing_tensor

    # Center coordinates
    centered_coords = coordinates - coordinates.mean(dim=0, keepdim=True)

    # Compute covariance matrix
    covariance_matrix = torch.matmul(centered_coords.T, centered_coords) / (coordinates.size(0) - 1)

    # Eigenvalue decomposition, sorted in ascending order
    eigenvalues = torch.linalg.eigvalsh(covariance_matrix)
    eigenvalues, _ = torch.sort(eigenvalues)

    # Calculate axis lengths and 2D diameters
    diameters = 4 * torch.sqrt(eigenvalues)
    max_2d_diameters = _calculate_max_2d_diameters(image, voxel_spacing)

    return eigenvalues, diameters, max_2d_diameters

def _calculate_max_2d_diameters(image, voxel_spacing):
    """
    Calculate maximum diameters in each 2D slice direction (column, row, slice).
    """
    coordinates = torch.nonzero(image > 0).float()
    if coordinates.numel() == 0:
        return torch.zeros(3, device=image.device)

    # Scale coordinates according to voxel spacing
    coordinates = coordinates[:, :3]
    voxel_spacing_tensor = torch.tensor(voxel_spacing, dtype=torch.float32, device=image.device)
    coordinates *= voxel_spacing_tensor

    # Calculate maximum distances in each 2D slice direction
    max_diameter_col = torch.max(coordinates[:, 1]) - torch.min(coordinates[:, 1])
    max_diameter_row = torch.max(coordinates[:, 2]) - torch.min(coordinates[:, 2])
    max_diameter_slice = torch.max(coordinates[:, 0]) - torch.min(coordinates[:, 0])
    return torch.tensor([max_diameter_col, max_diameter_row, max_diameter_slice], device=image.device)

def _calculate_sphericity(volume, surface_area):
    """
    Calculate sphericity based on volume and surface area.
    
    Args:
    volume (float): Volume of the shape.
    surface_area (float): Surface area of the shape.
    
    Returns:
    float: Sphericity value.
    """
    return (36 * torch.pi * volume ** 2) ** (1.0 / 3.0) / surface_area if surface_area > 0 else 0

def _calculate_compactness1(volume, surface_area):
    """
    Calculate Compactness 1 based on volume and surface area.
    
    Args:
    volume (float): Volume of the shape.
    surface_area (float): Surface area of the shape.
    
    Returns:
    float: Compactness 1 value.
    """
    return volume / (torch.sqrt(torch.pi * surface_area ** 3)) if surface_area > 0 else 0

def _calculate_compactness2(volume, surface_area):
    """
    Calculate Compactness 2 based on volume and surface area.
    
    Args:
    volume (float): Volume of the shape.
    surface_area (float): Surface area of the shape.
    
    Returns:
    float: Compactness 2 value.
    """
    return (36 * torch.pi * (volume ** 2)) / (surface_area ** 3) if surface_area > 0 else 0

def _calculate_flatness(eigenvalues):
    """
    Calculate flatness based on eigenvalues.
    
    Args:
    eigenvalues (torch.Tensor): A tensor of eigenvalues (sorted from smallest to largest).
    
    Returns:
    float: Flatness value.
    """
    if eigenvalues[-1] == 0:
        return 0.0
    return torch.sqrt(eigenvalues[0] / (eigenvalues[-1] + 1e-9)).item()

def _calculate_elongation(eigenvalues):
    """
    Calculate elongation based on eigenvalues.
    
    Args:
    eigenvalues (torch.Tensor): A tensor of eigenvalues (sorted from smallest to largest).
    
    Returns:
    float: Elongation value.
    """
    if eigenvalues[-1] == 0:
        return 0.0
    return torch.sqrt(eigenvalues[1] / (eigenvalues[-1] + 1e-9)).item()
