import torch
import torch.nn as nn

import torch.nn.functional as F
import torchvision.models as models
import torchvision.models.video as models_video
import json

class GlobalMaskedFeatureSelector(nn.Module):
    """
    Example CNN. You will need to adjust conv/pool layers to match your data shape
    so that the final linear layer has in_features= something -> 1842.
    """
    def __init__(self, in_channels=3, out_features=1824, dropout_prob=0.7, save_path="feature_importance.json"):
        super(GlobalMaskedFeatureSelector, self).__init__()
        self.save_path = save_path
        
        resnet3d = models_video.r3d_18(weights=models_video.R3D_18_Weights.DEFAULT)
        self.resnet = nn.Sequential(*list(resnet3d.children())[:-1])

        dummy_input = torch.randn(1, in_channels, 32, 128, 128)  # Simulated input
        with torch.no_grad():
            feature_dim = self.resnet(dummy_input).view(1, -1).size(1)  # Flatten and get size
        
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, out_features),
            nn.LayerNorm(out_features)
        )
        
        nn.init.constant_(self.fc[0].bias, -3.0)
        self.dropout = nn.Dropout(p=dropout_prob)


    def forward(self, x, return_logits: bool = False):
        """
        x shape: [B, in_channels=3, D=32, H=128, W=128].
        Returns: [B, out_features=1842].
        """
        x = self.resnet(x)  # Output shape: [B, C, 1, 1, 1]
        x = torch.flatten(x, start_dim=1)  # Shape: [B, C]
        x = self.dropout(x) 
        x = self.fc(x)  # Map to feature space [B, 1824]
        if return_logits:
            return x
        patient_specific_probs = torch.sigmoid(x)  # Element-wise sigmoid to yield probabilities in [0, 1]
        return patient_specific_probs

        
   
    
    def save_patient_feature_importances(self, patient_importances):
        """
        Saves a list of patient-specific feature importance dictionaries to JSON.
        
        patient_importances: a list of dictionaries. Each dictionary should have:
           - "patient_id": an identifier for the patient.
           - "feature_importance": list of importance values (length = out_features).
           - "sorted_indices": list of indices sorted in descending order of importance.
        """
        with open(self.save_path, 'w') as f:
            json.dump(patient_importances, f, indent=4)
        print(f"[INFO] Patient-specific feature importances saved to {self.save_path}")

    def load_feature_importance(self):
        try:
            with open(self.save_path, 'r') as f:
                data = json.load(f)
            print(f"[INFO] Feature importance loaded from {self.save_path}")
            return data
        except FileNotFoundError:
            print(f"[WARNING] No saved feature importance found at {self.save_path}")
            return None


class CNNWithGlobalMasking(nn.Module):
    def __init__(self, cnn_model, lr_model, dropout_prob=0.7, gating_config=None):
        """
        cnn_model:   A CNN that outputs a global mask [1, feature_dim].
        lr_model:    The logistic regression (InteractionLogisticRegression).
        """
        super(CNNWithGlobalMasking, self).__init__()
        self.cnn_model = cnn_model
        self.lr_model = lr_model
        self.gating_config = gating_config or {"enabled": False}
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, images_3d, radiomics_feats, gating_state=None, return_gating=False):
        """
        images_3d:      [B, 3, D, H, W]
        radiomics_feats:[B, 1842]
        Returns:        [B, 3] logits for multi-label classification
        """
        # 1) Generate the global mask
        gating_cfg = self.gating_config or {}
        use_gating = gating_cfg.get("enabled", False)
        if use_gating:
            feature_logits = self.cnn_model(images_3d, return_logits=True)
        else:
            feature_logits = self.cnn_model(images_3d)

        radiomics_feats = self.dropout(radiomics_feats)

        if feature_logits.shape[1] != radiomics_feats.shape[1]:
            raise ValueError(f"[ERROR] Mismatch: feature_importance.shape={feature_logits.shape}, expected radiomics_feats.shape[1]={radiomics_feats.shape[1]}")

        if use_gating:
            masked_feats, gating_outputs = self._apply_topm_gating(feature_logits, radiomics_feats, gating_state)
        else:
            feature_importance = torch.sigmoid(feature_logits)
            masked_feats = radiomics_feats * feature_importance
            gating_outputs = {
                "weights": feature_importance.detach(),
                "selected_mask": (feature_importance > 0).float().detach(),
            }

        logits = self.lr_model(masked_feats)
        if return_gating:
            return logits, gating_outputs
        return logits

    def _apply_topm_gating(self, logits, radiomics_feats, gating_state=None):
        from gating_utils import compose_continuous_topm_weights, hard_topk_mask, sample_gumbel

        cfg = self.gating_config
        gating_state = gating_state or {}
        tau_sched_default = cfg.get("tau_default", 1.0)
        tau_infer = cfg.get("tau_infer", cfg.get("tau_schedule", ((tau_sched_default, tau_sched_default),))[-1][-1] if cfg.get("tau_schedule") else 0.1)
        tau = gating_state.get("tau", tau_sched_default)
        lam = gating_state.get("lam", 0.0)
        stage = gating_state.get("stage", "inference")
        top_m = cfg.get("top_m", radiomics_feats.shape[1])
        weight_mode = cfg.get("weight_mode")
        if weight_mode is None:
            # Backward compatibility with the old boolean flag.
            weight_mode = "continuous" if cfg.get("use_continuous_weight_on_selected", False) else "binary"

        # Use a consistent temperature for the soft path.
        tau_eff = tau_infer if stage == "inference" else tau
        tau_eff = max(tau_eff, 1e-6)
        w_soft = torch.sigmoid(logits / tau_eff)

        if stage == "stage1":
            # Soft warm-up: no hard mask in the forward pass.
            weights = w_soft
            selected_mask = hard_topk_mask(logits, top_m).detach()
        elif stage == "stage2":
            g = sample_gumbel(logits.shape, device=logits.device) if lam > 0 else 0.0
            logits_noisy = logits + lam * g if lam > 0 else logits
            z_hard = hard_topk_mask(logits_noisy, top_m)
            if weight_mode == "continuous":
                weights, z_st = compose_continuous_topm_weights(z_hard, w_soft)
            else:
                z_st = z_hard + (w_soft - w_soft.detach())
                weights = z_st
            selected_mask = z_hard.detach()
        elif stage == "stage3":
            z_hard = hard_topk_mask(logits, top_m)
            if weight_mode == "continuous":
                weights, z_st = compose_continuous_topm_weights(z_hard, w_soft)
            else:
                z_st = z_hard + (w_soft - w_soft.detach())
                weights = z_st
            selected_mask = z_hard.detach()
        else:  # inference or any other stage treated as inference
            z_hard = hard_topk_mask(logits, top_m)
            if weight_mode == "continuous":
                weights = z_hard * w_soft
            else:
                weights = z_hard
            selected_mask = z_hard.detach()

        masked_feats = radiomics_feats * weights

        gating_outputs = {
            "weights": weights,
            "selected_mask": selected_mask,
            "tau": tau_eff,
            "lam": lam,
            "stage": stage,
        }
        return masked_feats, gating_outputs


    




