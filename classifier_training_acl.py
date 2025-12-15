
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import importlib

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as patches
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score

from reconstruction_model import UNet3D
#from diffusion_model import DDPM3D
from classifier import InteractionLogisticRegression
from feature_selector import GlobalMaskedFeatureSelector, CNNWithGlobalMasking



from dataset import MRIDataset, CombinedMRIFeatureDataset
from feature_extractor import first_order_and_shape_features
from torchradiomics_feature_extract import extract_all_radiomics
from sklearn.metrics import accuracy_score, recall_score

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_selection import VarianceThreshold
import torch.optim as optim
import json


# ------------------------------------------
# Radiomics-like Feature Extraction (1st order)
# ------------------------------------------

# -------------------------------------------------
# Feature Extraction Pipeline Using Pretrained DDPM
# -------------------------------------------------

import torch

def extract_features_with_pretrained_ddpm(
    loader,
    device,
    voxelArrayShift=0,
    pixelSpacing=[1.0, 1.0, 1.0]
):
    """
    Merged approach:
    
    1) For each batch of 3D volumes (potentially multi-channel, e.g. 3 views),
       corrupt the central region (e.g. 60% in each dimension).
    2) Add noise at a random diffusion timestep.
    3) Reconstruct using the pretrained diffusion model.
    4) Subdivide that same center region into 2x2x2=8 mini-patches.
    5) Extract first-order stats from each sub-patch (original & reconstructed),
       adding (loc_i, loc_j, loc_k) to each set of features.
    6) Return (all_features, all_labels).
    """

    all_features = []
    all_labels = []
    valid_indices = []

    with torch.no_grad():
        num_batches = len(loader)
        for batch_idx, (images, labels) in enumerate(loader):
            print(f"[DEBUG] Processing batch {batch_idx+1} / {num_batches}")
            images = images.to(device)  # shape: [B, C, D, H, W]
            labels = labels.to(device)  # shape: [B, ...] (depends on your dataset)

            B, C, D, H, W = images.shape
            print(f"    [DEBUG] Batch {batch_idx+1} shape: B={B}, C={C}, D={D}, H={H}, W={W}")

            batch_feats = []
            valid_batch_indices = []

            for b in range(B):
                sample_feats_all_views = []

                for view in range(C):
                    # -----------------------------------------
                    # 1) Corrupt the center region (60% in each dim)
                    # -----------------------------------------
                    single_view_image = images[b, view:view+1]  # shape: [1, D, H, W]
                    corrupted_images  = single_view_image.clone()

                    # Dimensions to corrupt (60% in each)
                    d = int(D * 0.5)
                    h = int(H * 0.3)
                    w = int(W * 0.5)

                    x = (D - d) // 2
                    y = (H - h) // 2
                    z = (W - w) // 2

                    # zero out center region
                    #corrupted_images[..., x:x + d, y:y + h, z:z + w] = 0

                    # -----------------------------------------
                    # 2) Add noise at random diffusion timestep
                    # -----------------------------------------
                    # We want a single random timestep for this single volume
                    #t = torch.randint(low=0, high=model.timesteps, size=(1,), device=device)
                    
                    # The model's add_noise() likely expects [B, C, D, H, W],
                    # so we add a batch dimension => shape [1, 1, D, H, W].
                    #corrupted_images_4d = corrupted_images.unsqueeze(0)  # [1, 1, D, H, W]
                    #x_t, _ = model.add_noise(corrupted_images_4d, t)      # [1, 1, D, H, W]

                    # -----------------------------------------
                    # 3) Reconstruct
                    # -----------------------------------------
                    #reconstructed_4d = model(x_t, t)  # [1, 1, D, H, W]
                    # remove the batch dim => shape [1, D, H, W]
                    #reconstructed = reconstructed_4d[0]

                    # -----------------------------------------
                    # 4) Subdivide the same center region into 2x2x2 mini-patches
                    #    and extract features from each patch (original & recon).
                    # -----------------------------------------
                    # We'll define slices for each dimension in 2 blocks:
                    #   block 0 -> first half
                    #   block 1 -> second half (including remainder if odd).
                    #
                    # For dimension D (depth):
                    #   block i_0 => [x : x + d//2]
                    #   block i_1 => [x + d//2 : x + d]
                    #
                    # Similar for H, W.

                    for i_ in range(2):
                        d_start = x + i_ * (d // 2)
                        d_end = x + d if i_ == 1 else (x + (i_ + 1) * (d // 2))

                        for j_ in range(2):
                            h_start = y + j_ * (h // 2)
                            h_end = y + h if j_ == 1 else (y + (j_ + 1) * (h // 2))

                            for k_ in range(2):
                                w_start = z + k_ * (w // 2)
                                w_end = z + w if k_ == 1 else (z + (k_ + 1) * (w // 2))

                                # Original mini-patch (shape: [1, d_sl, h_sl, w_sl])
                                original_patch = single_view_image[
                                    ..., d_start:d_end,
                                         h_start:h_end,
                                         w_start:w_end
                                ].float()
                            
                                if original_patch.sum() == 0 or torch.isnan(original_patch).any():
                                    print(f"[WARNING] Invalid patch at batch {batch_idx}, skipping.")
                                    continue

                                # 5) Extract first-order stats
                                orig_feats, _ = extract_all_radiomics(
                                    original_patch,
                                    voxelArrayShift=voxelArrayShift,
                                    pixelSpacing=pixelSpacing
                                )
                                #print(f"[INFO] Features extracted from subpatches before adding location: {len(orig_feats)}")
                                # Insert location as numeric features
                                orig_feats["loc_i"] = float(i_)
                                orig_feats["loc_j"] = float(j_)
                                orig_feats["loc_k"] = float(k_)

                                
                                for key, value in orig_feats.items():
                                    if isinstance(value, torch.Tensor) and torch.isnan(value).any():
                                        print(f"[ERROR] NaN in orig_feats at batch {batch_idx}, key={key}")
                                
                                subpatch_vec = []
                                for key in sorted(orig_feats.keys()):
                                    subpatch_vec.append(orig_feats[key])
                                
                                # Add sub-patch features to the entire channel's vector
                                sample_feats_all_views.extend(subpatch_vec)
                                #print(f"[INFO] Features extracted from subpatches: {len(orig_feats)}")

                
                
                if len(sample_feats_all_views) > 0:
                    sample_feats_tensor = torch.tensor(
                        sample_feats_all_views, dtype=torch.float, device=device
                    )
                    batch_feats.append(sample_feats_tensor)
                    valid_batch_indices.append(b)
                # Done with all channels for this sample -> convert to tensor, store
                

            if len(batch_feats) > 0:
                try:
                    batch_feats_tensor = torch.stack(batch_feats, dim=0)  # [B, feature_dim]
                    all_features.append(batch_feats_tensor)
                    all_labels.append(labels)
                    valid_indices.extend(valid_batch_indices)
                    print(f"    [DEBUG] Finished batch {batch_idx+1}, features shape = {batch_feats_tensor.shape}")
                except RuntimeError as e:
                    print(f"[ERROR] Failed to stack batch {batch_idx+1}: {e}")
            

    if len(all_features) > 0:
        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_features = torch.nan_to_num(all_features, nan=0.0, posinf=1e6, neginf=-1e6)
        return all_features, all_labels, valid_indices
    else:
        raise ValueError("[ERROR] No valid features were extracted.")# Concatenate across all batches
    

# -------------------------------------------------------------
# 3) Visualization Function: Original vs. Corrupted vs. Recon
# -------------------------------------------------------------
def visualize_reconstructions(
    diffusion_model,
    images,
    device,
    epoch,
    save_dir
):    
    """
    - Takes a small batch of images (3 views combined).
    - Processes each view independently through the diffusion model.
    - Plots [Original, Corrupted, Reconstructed] for the mid-slice of each view.
    - Saves figure to 'save_dir/epoch_{epoch}.png'.
    """
    diffusion_model.eval()
    with torch.no_grad():
        images = images.to(device)  # Shape: [B, 3, D, H, W]
        B, C, D, H, W = images.shape

        num_to_plot = min(5, B)

        for view in range(C):
            single_view_image = images[:, view:view+1, :, :, :]
            corrupted_images = single_view_image.clone()

            d, h, w = int(D * 0.5), int(H * 0.3), int(W * 0.5)
            x = (D - d) // 2
            y = (H - h) // 2
            z = (W - w) // 2
            corrupted_images[:, :, x:x + d, y:y + h, z:z + w] = 0

            t = torch.randint(low=0, high=diffusion_model.timesteps, size=(B,), device=device)
            x_t, _ = diffusion_model.add_noise(corrupted_images, t)

            # Reconstruct the images
            reconstructed = diffusion_model(x_t, t)

            for b in range(num_to_plot):
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                # Display original image
                axes[0].imshow(single_view_image[b, 0, D // 2, :, :].cpu(), cmap='gray')
                axes[0].set_title("Original")
                axes[0].axis("off")

                # Display corrupted image
                axes[1].imshow(corrupted_images[b, 0, D // 2, :, :].cpu(), cmap='gray')
                axes[1].set_title("Corrupted")
                axes[1].axis("off")

                # Display reconstructed image
                axes[2].imshow(reconstructed[b, 0, D // 2, :, :].cpu(), cmap='gray')
                axes[2].set_title("Reconstructed")
                axes[2].axis("off")

                # Save the figure
                view_dir = os.path.join(save_dir, f"view_{view+1}")
                os.makedirs(view_dir, exist_ok=True)
                save_path = os.path.join(view_dir, f"epoch_{epoch}_image_{b+1}.png")
                plt.savefig(save_path, bbox_inches='tight')
                plt.close(fig)

    print(f"[DEBUG] Visualization for epoch {epoch} saved in {save_dir}")


# ----------------------------------------------------------------
# Metrics: Sensitivity, Specificity, AUC (per label, then avg)
# ----------------------------------------------------------------

def compute_metrics(probabilities, labels):
    """
    Computes metrics for ACL binary classification.

    probabilities: shape [N,1], each in [0,1]
    labels:        shape [N,1], each 0 or 1

    Returns a dictionary with:
      - Accuracy
      - Sensitivity (Recall)
      - Specificity
      - AUC
      - F1-score
    """
    probabilities_np = probabilities.detach().cpu().numpy().flatten()  # [N]
    labels_np = labels.detach().cpu().numpy().flatten()  # [N]

    from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, accuracy_score

    # Convert probabilities to binary predictions (threshold = 0.5)
    preds = (probabilities_np >= 0.5).astype(int)

    # Compute Accuracy
    accuracy = accuracy_score(labels_np, preds)

    # Compute Confusion Matrix values
    tn, fp, fn, tp = confusion_matrix(labels_np, preds, labels=[0,1]).ravel()

    # Compute Sensitivity (Recall)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # Compute Specificity (True Negative Rate)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Compute AUC
    try:
        auc = roc_auc_score(labels_np, probabilities_np)
    except ValueError:
        auc = 0.0  # Handle cases where only one class is present

    # Compute F1-score
    f1 = f1_score(labels_np, preds, zero_division=0)

    # Print results
    print(f"ACL Metrics: Acc={accuracy:.3f}, Sensitivity={sensitivity:.3f}, "
          f"Specificity={specificity:.3f}, AUC={auc:.3f}, F1={f1:.3f}")

    # Return metrics in a dictionary
    metrics = {
        "label_name": "ACL",
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "auc": auc,
        "f1": f1
    }

    return metrics




# ----------------------------------------------------------------
# Simple classification training for your 3-class prediction
# ----------------------------------------------------------------
def find_best_thresholds(probabilities, labels, label_names=None, min_threshold=0.25):
    """
    Finds the best threshold based on a combination of accuracy and sensitivity (recall).
    
    Parameters:
    probs:  1D numpy array of predicted probabilities for one label
    truths: 1D numpy array of 0/1 for that same label
    label_name: Name of the label (for logging purposes)

    Returns:
    best_thresh: The threshold that optimizes the combined metric
    """

    if label_names is None:
        label_names = ["ACL"]

    probabilities_np = probabilities.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()

    best_thresholds = []
    
    for i, lbl_name in enumerate(label_names):
        best_thr = min_threshold
        best_f1 = 0.0

        for thr in np.linspace(min_threshold, 1, int((1 - min_threshold) * 100) + 1):
            preds_i = (probabilities_np[:, i] >= thr).astype(int)
            truths_i = labels_np[:, i]
            f1_i = f1_score(truths_i, preds_i, zero_division=0)

            if f1_i > best_f1:
                best_f1 = f1_i
                best_thr = thr

        best_thresholds.append(best_thr)
        print(f"Label '{lbl_name}': best threshold = {best_thr:.2f}, F1 = {best_f1:.3f}")

    return best_thresholds

def find_best_thresholds_youden(probabilities, labels, min_threshold=0.1):
    """
    Finds the best threshold for ACL classification using Youden's J statistic.
    
    probabilities: Tensor of shape [N,1] - predicted probabilities for ACL
    labels:        Tensor of shape [N,1] - ground truth binary labels for ACL
    
    Returns:
    best_thr: The threshold that maximizes Youden's J statistic
    """
    probabilities_np = probabilities.detach().cpu().numpy().flatten()  # Convert to 1D array
    labels_np = labels.detach().cpu().numpy().flatten()                # Convert to 1D array

    best_thr = min_threshold
    best_j = -1

    # Iterate over possible thresholds from min_threshold to 1.0
    for thr in np.linspace(min_threshold, 1, int((1 - min_threshold) * 100) + 1):
        preds = (probabilities_np >= thr).astype(int)

        # Calculate confusion matrix values
        tn, fp, fn, tp = confusion_matrix(labels_np, preds, labels=[0, 1]).ravel()

        # Calculate sensitivity and specificity
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0

        # Youden's J statistic
        j_stat = sens + spec - 1

        # Update the best threshold based on J statistic
        if j_stat > best_j:
            best_j = j_stat
            best_thr = thr

    print(f"Best threshold for ACL = {best_thr:.2f}, Youden's J = {best_j:.3f}")

    return best_thr



def train_cnn_with_masking(
    model, 
    train_loader, 
    val_loader, 
    device='cuda',
    num_epochs=10000,
    lr=1e-3,
    weight_decay=1e-4,
    patience=1000,
    model_save_path='combined_model_acl.pth',
    update_threshold_every=5,
    feature_importance_path='feature_importance.json',
    save_every=20
):
    """
    model: CNNWithFeatureMasking
    train_loader: yields (images_3d, feats_1824, labels_3)
    val_loader: same
    """
    
    train_labels_full = []

    for _, _, labels in train_loader:
        train_labels_full.append(labels[:, 1].unsqueeze(1))  # Extract only ACL labels
        

    train_labels_full = torch.cat(train_labels_full, dim=0)  # shape [N,1]
    
    pos_count = train_labels_full.sum().item()  # Number of positive ACL cases
    neg_count = len(train_labels_full) - pos_count  # Number of negative ACL cases

    pos_weight = neg_count / (pos_count + 1e-8)  # Avoid division by zero
    pos_weight = np.clip(pos_weight, 0.35, 2.75)  # Limit range

    pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float, device=device)
    print(f"[DEBUG] ACL pos_weight = {pos_weight:.3f}")


    model = model.to(device)
    # Define the loss function with class weights
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)  # Increase weight for minority class



    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=5e-4, steps_per_epoch=len(train_loader), epochs=num_epochs)



    best_val_f1 = 0.0
    epochs_no_improve = 0
    best_thresholds = 0.5

    print("Starting combined CNN+LR training...")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_train_probs = []
        all_train_labels = []

        for images_3d, feats_1824, labels in train_loader:
            images_3d = images_3d.to(device)     # [B, 3, D, H, W]
            feats_1824 = feats_1824.to(device)   # [B, 1842]
            #feats_1824 = feats_1824[:, selected_features_mask]
            feats_1824 = torch.tensor(feats_1824, dtype=torch.float32, device=device)  # Convert back to Tensor

            labels = labels.to(device)[:, 1].unsqueeze(1)  # Extract ACL labels
            
            #print(f"[DEBUG] Radiomics Features - Mean: {torch.mean(feats_1824).item():.6f}")
            #print(f"[DEBUG] Radiomics Features - Std: {torch.std(feats_1824).item():.6f}")
            #print(f"[DEBUG] Radiomics Features - Min: {torch.min(feats_1824).item():.6f}, Max: {torch.max(feats_1824).item():.6f}")
            
            # Check for NaNs or Infs
            if torch.isnan(feats_1824).any():
                print("[ERROR] NaN values detected in radiomics_feats!")
            if torch.isinf(feats_1824).any():
                print("[ERROR] Infinite values detected in radiomics_feats!")
            
            #feature_variability = torch.mean(torch.std(feats_1824, dim=0))
            #print(f"[DEBUG] Radiomics Feature Variability: {feature_variability.item():.6f}")

            smoothed_labels = labels * 0.9 + 0.05  

            optimizer.zero_grad()



            logits = model(images_3d, feats_1824)  # => [B, 3]
            #print(f"[DEBUG] First 10 Raw Model Outputs: {logits[:10].detach().cpu().numpy()}")
            loss = criterion(logits, smoothed_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Compute training probabilities
            probs = torch.sigmoid(logits).detach().cpu()
            all_train_probs.append(probs)
            all_train_labels.append(labels.cpu())

        train_loss = running_loss / len(train_loader)

        all_train_probs = torch.cat(all_train_probs, dim=0)
        all_train_labels = torch.cat(all_train_labels, dim=0)

        train_preds = (all_train_probs >= best_thresholds).float()
        train_metrics = compute_metrics(train_preds, all_train_labels)

        if epoch == 0 or epoch % 10 == 0:
            with torch.no_grad():
                # Get one batch from the validation loader.
                for images_3d_val, feats_1824_val, labels_val in val_loader:
                    images_3d_val = images_3d_val.to(device)
                    # Obtain patient-specific importance vectors from the CNN branch.
                    patient_feature_importance = model.cnn_model(images_3d_val)  # shape: [B, out_features]
                    num_samples = min(2, patient_feature_importance.shape[0])
                    for patient_idx in range(num_samples):
                        feature_vector = patient_feature_importance[patient_idx]  # shape: [out_features]
                        feature_vector_np = feature_vector.cpu().numpy()
                        sorted_indices = np.argsort(feature_vector_np)[::-1]  # descending order
                        sorted_importances = feature_vector_np[sorted_indices]

                        print(f"\n[INFO] Epoch {epoch+1}: Validation Sample {patient_idx+1} Top 10 Most Important Features:")
                        for i in range(10):
                            print(f"    Rank {i+1}: Feature {sorted_indices[i]} → Importance = {sorted_importances[i]:.6f}")

                        print(f"\n[INFO] Epoch {epoch+1}: Validation Sample {patient_idx+1} 10 Least Important Features:")
                        for i in range(10):
                            print(f"    Rank {len(sorted_importances) - 10 + i + 1}: Feature {sorted_indices[-(10-i)]} → Importance = {sorted_importances[-(10-i)]:.6f}")
                    # Only use the first batch for logging.
                    break

        # Validation
        model.eval()
        val_loss = 0.0
        

        all_val_probs = []
        all_val_labels = []

        with torch.no_grad():
            for images_3d, feats_1824, labels in val_loader:
                images_3d = images_3d.to(device)
                feats_1824 = feats_1824.cpu().numpy()  # Convert tensor to NumPy

                # Apply VarianceThreshold mask to validation set
                #feats_1824 = feats_1824[:, selected_features_mask]
                feats_1824 = torch.tensor(feats_1824, dtype=torch.float32, device=device)
                labels = labels.to(device)[:, 1].unsqueeze(1)
                #print(f"[DEBUG] Validation Features - Mean: {torch.mean(feats_1824).item():.6f}")
                #print(f"[DEBUG] Validation Features - Std: {torch.std(feats_1824).item():.6f}")
                #print(f"[DEBUG] Validation Features - Min: {torch.min(feats_1824).item():.6f}, Max: {torch.max(feats_1824).item():.6f}")


                logits = model(images_3d, feats_1824)
                loss = criterion(logits, labels)
                val_loss += loss.item()

                probs = torch.sigmoid(logits)
                all_val_probs.append(probs.cpu())
                all_val_labels.append(labels.cpu())

        val_loss /= len(val_loader)
        


        all_val_probs  = torch.cat(all_val_probs,  dim=0)
        all_val_labels = torch.cat(all_val_labels, dim=0)
        
        #if epoch % save_every == 0:
            #model.cnn_model.save_feature_importance()
            #print(f"[INFO] Feature importance saved at epoch {epoch+1}")
        # Use your existing compute_metrics function
        if epoch % update_threshold_every == 0 or epoch == 0:
            best_thresholds = find_best_thresholds_youden(all_val_probs, all_val_labels)
            print(f"[INFO] Updated thresholds at epoch {epoch+1}: {best_thresholds}")
        
        thresholds_tensor = torch.tensor(best_thresholds, device=device)
        val_preds = (all_val_probs.to(device) >= thresholds_tensor).long()
        val_metrics = compute_metrics(val_preds, all_val_labels)

        # Mean F1 scores
        train_f1 = float(train_metrics["f1"])

        val_f1 = float(val_metrics["f1"])


        if (epoch + 1) % 1 == 0 or epoch == 0:  # print every epoch (or modify as you wish)
            print(f"Epoch [{epoch+1}/{num_epochs}]")
            print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"  Train Mean F1: {train_f1:.3f} | Val Mean F1: {val_f1:.3f}")

            print("\n  Training Metrics:")
            print(f"    ACL: "
                  f"Acc={train_metrics['accuracy']:.2f} "
                  f"Sens={train_metrics['sensitivity']:.2f} "
                  f"Spec={train_metrics['specificity']:.2f} "
                  f"AUC={train_metrics['auc']:.2f} "
                  f"F1={train_metrics['f1']:.2f}")
            
                
            print("\n  Validation Metrics:")
            print(f"    ACL: "
                  f"Acc={val_metrics['accuracy']:.2f} "
                  f"Sens={val_metrics['sensitivity']:.2f} "
                  f"Spec={val_metrics['specificity']:.2f} "
                  f"AUC={val_metrics['auc']:.2f} "
                  f"F1={val_metrics['f1']:.2f}")

            print("")

        #if hasattr(model, 'cnn_model') and hasattr(model.cnn_model, 'global_mask'):
            #global_mask = model.cnn_model.global_mask.detach().cpu().numpy()
            #print(f"[DEBUG] Global mask stats: mean={global_mask.mean():.4f}, "
                  #f"std={global_mask.std():.4f}, min={global_mask.min():.4f}, max={global_mask.max():.4f}")
        # Ensure scheduler updates even if loss stagnates
        if val_loss >= best_val_f1:
            scheduler.step(val_loss + 1e-6)  # Small epsilon to ensure step is called
        else:
            scheduler.step(val_loss)

        min_epochs = 50
        if epoch >= min_epochs and val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"  [*] New best val F1: {best_val_f1:.3f}. Saved model to {model_save_path}")
        else:
            epochs_no_improve += 1
            print(f"  [*] No improvement in val F1 for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= patience:
            print(f"  [*] Early stopping triggered at epoch {epoch+1}.")
            break

        
        if epoch >= min_epochs and val_f1 >= 0.98:
            print(f"  [*] Early stopping triggered at epoch {epoch+1} "
                  f"because Val MeanF1 reached {val_f1:.3f} >= 0.98.")
            # Make sure we save the model again if this is indeed the best so far.
            if val_f1 > best_val_f1:
                torch.save(model.state_dict(), model_save_path)
            break

    print("Loading best model weights for threshold tuning...")
    model.load_state_dict(torch.load(model_save_path))
    model.eval()

    all_probs_val = []
    all_labels_val = []
    with torch.no_grad():
        for images_3d, feats_1824, labels in val_loader:
            images_3d = images_3d.to(device)
            feats_1824 = feats_1824.to(device)
            logits = model(images_3d, feats_1824)
            probs = torch.sigmoid(logits).cpu()
            all_probs_val.append(probs)
            all_labels_val.append(labels[:, 1].unsqueeze(1).cpu())

    all_probs_val  = torch.cat(all_probs_val,  dim=0)
    all_labels_val = torch.cat(all_labels_val, dim=0)

    # Find best thresholds
    #label_names = ["abnormal"]
    best_thresholds = find_best_thresholds_youden(all_probs_val, all_labels_val)

    print("Done with training and threshold selection.")
    print("Best thresholds per label:", best_thresholds)

    return model, best_thresholds





def evaluate_model(model, loader, device, thresholds=None, output_log_file=None):
    model.eval()
    all_probs  = []
    all_labels = []

    with torch.no_grad():
        for images_3d, feats_1824, labels_3 in loader:
            images_3d  = images_3d.to(device)
            feats_1824 = feats_1824.to(device)
            labels_3   = labels_3.to(device)

            logits = model(images_3d, feats_1824)  # => [B,3]
            probs  = torch.sigmoid(logits)

            all_probs.append(probs)  # Keep on the same device as `probs`
            all_labels.append(labels_3)

    all_probs  = torch.cat(all_probs,  dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    if thresholds is not None:
        thresholds_tensor = torch.tensor(thresholds, device=device)

        preds = (all_probs >= thresholds_tensor).long()  # Use best thresholds
    else:
        preds = (all_probs >= 0.5).long()  # Default threshold is 0.5

    # Compute labelwise metrics
    metrics_dict = compute_metrics(preds, all_labels)
    labelwise_acc  = metrics_dict["labelwise_accuracy"]
    labelwise_sens = metrics_dict["labelwise_sensitivity"]
    labelwise_spec = metrics_dict["labelwise_specificity"]
    labelwise_auc  = metrics_dict["labelwise_auc"]

    # Mean ± std across 3 labels
    acc_mean,  acc_std  = np.mean(labelwise_acc),  np.std(labelwise_acc)
    sens_mean, sens_std = np.mean(labelwise_sens), np.std(labelwise_sens)
    spec_mean, spec_std = np.mean(labelwise_spec), np.std(labelwise_spec)
    auc_mean,  auc_std  = np.mean(labelwise_auc),  np.std(labelwise_auc)

    # Print label-wise
    output = []
    output.append("Evaluation Results (Label-wise):")
    label_names = metrics_dict["label_names"]
    for i, lbl_name in enumerate(label_names):
        output.append(f"  {lbl_name:9s} | "
                      f"Acc={labelwise_acc[i]:.3f} | "
                      f"Sens={labelwise_sens[i]:.3f} | "
                      f"Spec={labelwise_spec[i]:.3f} | "
                      f"AUC={labelwise_auc[i]:.3f}")

    output.append("\nOverall (mean ± std across 3 labels):")
    output.append(f"  Accuracy:    {acc_mean:.3f} ± {acc_std:.3f}")
    output.append(f"  Sensitivity: {sens_mean:.3f} ± {sens_std:.3f}")
    output.append(f"  Specificity: {spec_mean:.3f} ± {spec_std:.3f}")
    output.append(f"  AUC:         {auc_mean:.3f} ± {auc_std:.3f}\n")

    # Print and save to file
    if output_log_file:
        with open(output_log_file, 'w') as f:
            f.write("\n".join(output))
        print(f"Results saved to {output_log_file}")
    else:
        print("\n".join(output))

    results_summary = {
        "acc_mean": acc_mean, "acc_std": acc_std,
        "sens_mean": sens_mean, "sens_std": sens_std,
        "spec_mean": spec_mean, "spec_std": spec_std,
        "auc_mean": auc_mean, "auc_std": auc_std
    }

    return metrics_dict, results_summary

def check_for_nans_or_infs(tensor, name):
    if torch.isnan(tensor).any():
        print(f"[ERROR] {name} contains NaN values.")
    if torch.isinf(tensor).any():
        print(f"[ERROR] {name} contains Inf values.")

# -----------------------
# Main Script Example
# -----------------------
if __name__ == "__main__":
    """
    Example usage:
      1. Instantiate your MRIDataset for 'train' phase.
      2. Load the pretrained DDPM model.
      3. Extract features by corrupting + reconstructing.
      4. Train LesionClassifier using those features.
    
    Adjust the paths, CSV files, and phases for your real environment.
    """

    from sklearn.model_selection import train_test_split
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    #model_path = '/home/yaxi/HealKnee_ddpm_central/best_diffusion_model.pth'

    #diffusion_model = DDPM3D().to(device)
    #checkpoint = torch.load(model_path, map_location=device)
    #diffusion_model.load_state_dict(checkpoint)
    #diffusion_model.eval()
    #print(f"Loaded pretrained DDPM3D model from: {model_path}")

    visualization_save_path = '/home/yaxi/miccai_workshop'
    os.makedirs(visualization_save_path, exist_ok=True)
    
    model_save_path = '/home/yaxi/miccai_workshop'
    os.makedirs(model_save_path, exist_ok=True)   

    features_path = os.path.join(model_save_path, "features_1824_acl_clinical.pt")
    labels_path = os.path.join(model_save_path, "labels_3_acl_clinical.pt")
    norm_stats_path = os.path.join(model_save_path, "normalization_stats_acl_clinical.json")

    feature_importance_path = os.path.join(model_save_path, "feature_importance_acl_clinical.json")

    # 1) Create your training dataset & loader
    root_dir = "/home/yaxi/MRNet-v1.0_nii"
    labels_files = {
        'abnormal': os.path.join(root_dir, 'train-abnormal.csv'),
        'acl': os.path.join(root_dir, 'train-acl.csv'),
        'meniscus': os.path.join(root_dir, 'train-meniscus.csv')
        }
    phase = 'train'  # or 'valid', 'test', etc.

    train_mri_dataset = MRIDataset(
        root_dir=root_dir,
        labels_files=labels_files,
        phase=phase,
        views=('coronal_reg', 'axial_reg', 'sagittal_reg'),      
        transform=None,
        target_size=(32, 128, 128)   
    )
    
    labels_files_test = {
        'abnormal': os.path.join(root_dir, 'valid-abnormal.csv'),
        'acl':      os.path.join(root_dir, 'valid-acl.csv'),
        'meniscus': os.path.join(root_dir, 'valid-meniscus.csv')
    }

    test_mri_dataset = MRIDataset(
        root_dir=root_dir,
        labels_files=labels_files_test,
        phase='valid',
        views=('coronal_reg', 'axial_reg', 'sagittal_reg'),
        transform=None,
        target_size=(32, 128, 128)
    )
    print(f"[INFO] Created test dataset with {len(test_mri_dataset)} samples.")


    #train_loader_for_ddpm = DataLoader(
    #    torch.utils.data.Subset(train_mri_dataset, range(50)),
    #    batch_size=8,
    #    shuffle=False
    #)

    train_loader_for_ddpm = DataLoader(train_mri_dataset, batch_size=8, shuffle=False)
    print("Total MRI samples:", len(train_mri_dataset))

    if os.path.exists(features_path) and os.path.exists(labels_path):
        print("[INFO] Loading saved radiomics features and labels...")
        features_1824 = torch.load(features_path, map_location='cuda:1')
        labels_3      = torch.load(labels_path, map_location='cuda:1')

        print(f"[INFO] Loaded features shape: {features_1824.shape}")
        print(f"[INFO] Loaded labels shape: {labels_3.shape}")
        valid_indices = list(range(len(features_1824)))
    else:
        print("[INFO] Extracting radiomics features from scratch...")
        features_1824, labels_3, valid_indices = extract_features_with_pretrained_ddpm(
            loader=train_loader_for_ddpm,
            device=device,
            voxelArrayShift=0,
            pixelSpacing=[1.0, 1.0, 1.0]
        )
        print(f"[INFO] Extracted features shape: {features_1824.shape}")
        print(f"[INFO] Extracted labels shape: {labels_3.shape}")
        

        # Save the extracted features and labels
        print("[INFO] Saving radiomics features and labels...")
        torch.save(features_1824, features_path)
        torch.save(labels_3, labels_path)
        print(f"[INFO] Features saved to: {features_path}")
        print(f"[INFO] Labels saved to: {labels_path}")

    #check_for_nans_or_infs(features_1824, "features_1824")
    #check_for_nans_or_infs(labels_3, "labels_3")
    means = features_1824.mean(dim=0, keepdim=True)
    stds  = features_1824.std(dim=0, keepdim=True)
    features_1824 = (features_1824 - means) / (stds + 1e-8)
    #check_for_nans_or_infs(features_1824, "features_1824")

    norm_stats = {
        "means": means.squeeze().tolist(),
        "stds": stds.squeeze().tolist()
    }
    if not os.path.exists(norm_stats_path):
        with open(norm_stats_path, 'w') as f:
            json.dump(norm_stats, f, indent=4)
        print(f"[INFO] Normalization stats saved to: {norm_stats_path}")
    else:
        print(f"[INFO] Normalization stats already exist: {norm_stats_path}")

    train_mri_dataset = torch.utils.data.Subset(train_mri_dataset, valid_indices)
    print(f"[INFO] Filtered dataset length: {len(train_mri_dataset)}")
    combined_dataset = CombinedMRIFeatureDataset(train_mri_dataset, features_1824)

    labels_acl = labels_3[:, 1].unsqueeze(1) 

    train_indices, val_indices = train_test_split(
        np.arange(len(labels_acl)),
        test_size=0.3,  # 25% for validation
        stratify=labels_acl.cpu().numpy(),
        random_state=42
    )

    # Create subset datasets
    train_ds = torch.utils.data.Subset(combined_dataset, train_indices)
    val_ds = torch.utils.data.Subset(combined_dataset, val_indices)

 
    
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    #train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    #val_loader   = DataLoader(val_ds, batch_size=8, shuffle=False)
    #mask_file_path = os.path.join(model_save_path, "selected_features_mask_men.npy")
    feature_importance_path = os.path.join(model_save_path, "feature_importance_acl_clinical.json")

# Extract features from the training dataset
    all_features = []
    for _, feats_1824, _ in train_loader:
        all_features.append(feats_1824.cpu().numpy()) 

# Convert list of NumPy arrays into a single 2D array
    all_features_np = np.vstack(all_features)


    cnn_model = GlobalMaskedFeatureSelector(in_channels=3, out_features=2568, save_path = feature_importance_path)

    lr_model  = InteractionLogisticRegression(input_size=2568, output_size=1)

    combined_model = CNNWithGlobalMasking(cnn_model=cnn_model, lr_model=lr_model)
    combined_model = combined_model.to(device)


    best_model_path = os.path.join(model_save_path, "best_cnn_lr_mask_model_softmax_2*2_acl_clinical.pth")
    output_log_file = os.path.join(model_save_path, "evaluation_metrics_softmax_2*2_acl_clinical.txt")

    trained_model, best_thresholds = train_cnn_with_masking(
        model=combined_model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=1000,  # Set your desired number of epochs
        lr=1e-4,
        weight_decay=1e-3,
        patience=150,
        model_save_path=best_model_path,
        feature_importance_path=feature_importance_path
    )
    print("Done!")
    trained_model.cnn_model.save_feature_importance()
    print("[INFO] Feature importance successfully saved after training.")

    thresholds_norm_path = os.path.join(model_save_path, "thresholds_and_normalization_softmax_2*2_acl_clinical.json")

    if isinstance(best_thresholds, (np.float64, float)):
        best_thresholds = [float(best_thresholds)]
    else:
        best_thresholds = [float(thr) for thr in best_thresholds]

    # Save best thresholds, means, and stds to a JSON file
    thresholds_and_norm = {
        "thresholds": best_thresholds,  # JSON serializable
        "means": means.squeeze().tolist(),  # Convert tensor to list
        "stds": stds.squeeze().tolist()  # Convert tensor to list 
    }

    with open(thresholds_norm_path, 'w') as f:
        json.dump(thresholds_and_norm, f, indent=4)
    print(f"[INFO] Saved thresholds and normalization stats to {thresholds_norm_path}")

    


