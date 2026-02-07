from pathlib import Path
import torch
from torch import nn, Tensor
from typing import Optional, Literal
from prt_rl.common.networks.cnn import CNNNetwork
from prt_rl.common.specs.networks import CNNNetworkSpec

from prt_rl.model_based.models.representation.interface import RepresentationInterface


class KeypointModel(RepresentationInterface):
    """
    Keypoint encoder interface: images -> keypoints (+ optional heatmaps/features).

    Intended to support:
      - Self-supervised keypoint discovery (heatmaps + soft-argmax)
      - Downstream control (q, dq from keypoints)
      - MPC pipelines (state extraction + diagnostics)
    """

    def __init__(self,
                 num_features: int,
                 num_keypoints: int,
                 backbone: nn.Module,
                 feature_head: nn.Module,
                 keypoint_head: nn.Module,
                 decoder: nn.Module,
                 normalize_coords: bool = False,
                 ):
        super().__init__()
        self.D = num_features
        self.K = num_keypoints
        self.backbone = backbone
        self.feature_head = feature_head
        self.keypoint_head = keypoint_head
        self.decoder = decoder
        self.normalize_coords = normalize_coords

    # ----------------------------
    # Core API
    # ----------------------------
    # def forward(self, obs: Tensor) -> Dict[str, Tensor]:
    #     """
    #     Default forward pass.

    #     Args:
    #         obs: image tensor, shape [B, C, H, W] (or [B, H, W, C] if you support channels_last)
    #     Returns:
    #         dict that should minimally include:
    #           - "keypoints": [B, K, 2]  (coordinate space depends on cfg.normalize_coords)
    #         and may include optional keys:
    #           - "heatmaps":  [B, K, H', W']
    #           - "scores":    [B, K, H', W'] (pre-softmax logits)
    #           - "features":  [B, D, H', W']
    #           - "confidence":[B, K] or [B, K, 1]
    #     """
    #     features = self.backbone(obs)
    #     scores = self.keypoint_head(features)
    #     return scores

    def encode(self, obs: Tensor) -> Tensor:
        """
        Convenience wrapper to return keypoints only.

        Returns:
            keypoints: [B, K, 2]
        """
        scores = self.score_maps(obs)
        probs = self.spatial_softmax(scores)
        return self.soft_argmax(probs, normalize=self.normalize_coords)
    
    def save(self, path: str | Path) -> None:
        # Save backbone, keypoint head, feature head, detector parameters
        pass

    def load(cls, path: str | Path, map_location: str | torch.device = "cpu") -> "RepresentationInterface":
        pass
    
    @staticmethod
    def _build_decoder_network(
        backbone_spec: "CNNNetworkSpec",
        *,
        in_channels: int,                              # D (feature channels)
        out_channels: int = 3,                          # RGB
        out_activation: Optional[Literal["sigmoid", "tanh"]] = None,
        upsample: Literal["nearest", "bilinear"] = "nearest",
    ) -> nn.Module:
        """
        Build a decoder that inverts the backbone_spec's spatial downsampling.

        Assumptions / intent:
        - backbone_spec describes a feature-map backbone (pooling should be None for Transporter).
        - The decoder maps transported features [B, D, H', W'] -> image [B, out_channels, H, W]
            where (H, W) == backbone_spec.input_hw.
        - Uses Upsample(+Conv) blocks (stable; avoids deconv checkerboarding).
        - Upsampling factors are derived from backbone_spec.strides, in reverse order.

        Notes:
        - This method does NOT need to know the actual runtime H',W' if you are guaranteed that
            the transported features already match the backbone output spatial size.
        - If strides include values > 2, this will upsample by that factor directly.

        Returns:
        nn.Module (typically nn.Sequential)
        """
        if backbone_spec.input_hw is None:
            raise ValueError("backbone_spec.input_hw must be set to build a decoder with correct output resolution.")
        if backbone_spec.pooling is not None:
            raise ValueError("For Transporter-style feature maps, backbone_spec.pooling should be None.")

        if not (len(backbone_spec.channels) == len(backbone_spec.kernel_sizes) ==
                len(backbone_spec.strides) == len(backbone_spec.paddings)):
            raise ValueError("Backbone spec channels/kernel_sizes/strides/paddings must have same length.")

        # We'll invert the stride schedule: iterate from last conv back to first.
        rev_strides = list(backbone_spec.strides)[::-1]
        rev_channels = list(backbone_spec.channels)[::-1]

        layers: list[nn.Module] = []

        prev_c = in_channels

        # Heuristic channel plan:
        # After each upsample stage, map channels toward earlier backbone widths (reverse order).
        # Example backbone channels: [32, 64, 64] -> rev_channels: [64, 64, 32]
        # Start from D -> 64 -> 64 -> 32 -> ... -> out_channels
        ch_idx = 0

        for s in rev_strides:
            # Optionally upsample if the corresponding encoder step downsampled
            if s != 1:
                if upsample == "nearest":
                    layers.append(nn.Upsample(scale_factor=s, mode="nearest"))
                elif upsample == "bilinear":
                    layers.append(nn.Upsample(scale_factor=s, mode="bilinear", align_corners=False))
                else:
                    raise ValueError(f"Unknown upsample mode: {upsample}")

            # Choose an output channel width for this stage.
            # We advance through reversed backbone channels to "mirror" encoder widths.
            out_c = rev_channels[min(ch_idx, len(rev_channels) - 1)]
            ch_idx += 1

            # A basic conv block
            layers.append(nn.Conv2d(prev_c, out_c, kernel_size=3, stride=1, padding=1))
            layers.append(_norm_layer(backbone_spec.norm, out_c, backbone_spec.group_norm_groups))
            layers.append(_activation(backbone_spec.activation))
            if backbone_spec.dropout and backbone_spec.dropout > 0.0:
                layers.append(nn.Dropout2d(p=backbone_spec.dropout))

            prev_c = out_c

        # Final RGB (or out_channels) projection
        layers.append(nn.Conv2d(prev_c, out_channels, kernel_size=3, stride=1, padding=1))
        layers.append(_output_activation(out_activation))

        return nn.Sequential(*layers)
    
    # ----------------------------
    # Detector outputs (building blocks)
    # ----------------------------
    def score_maps(self, obs: Tensor) -> Tensor:
        """
        Return raw keypoint score/logit maps before spatial softmax.

        Returns:
            scores: [B, K, H', W']
        """
        latent_features = self.backbone(obs)
        S_x = self.keypoint_head(latent_features)
        return S_x

    def spatial_softmax(self, S_x: Tensor) -> Tensor:
        """
        Apply per-keypoint spatial softmax to produce distributions.

        Args:
            S_x: [B, K, H', W']
        Returns:
            probs:  [B, K, H', W']
        """
        # Compute the spatial softmax probabilities per keypoint
        B, K, H, W = S_x.shape
        flat = S_x.view(B, K, H * W)    # [B, K, H*W]
        probs_flat = torch.softmax(flat, dim=-1)  # softmax over H*W
        P_x = probs_flat.view(B, K, H, W)  # [B, K, H, W]
        return P_x

    def soft_argmax(self, P_x: Tensor, normalize: bool = False) -> Tensor:
        """
        Convert spatial distributions to keypoint coordinates via expectation.

        Args:
            P_x: [B, K, H', W']
            normalize: whether to normalize coordinates to [-1, 1] range
        Returns:
            Phi_x: [B, K, 2]
        """
        B, K, H, W = P_x.shape
        device, dtype = P_x.device, P_x.dtype

        if normalize:
            u = torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype) # [W]
            v = torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype) # [H]
        else:
            u = torch.linspace(0.0, W - 1, W, device=device, dtype=dtype)
            v = torch.linspace(0.0, H - 1, H, device=device, dtype=dtype)

        uu, vv = torch.meshgrid(u, v, indexing='xy')  # [W], [H] -> [H, W]

        x = (P_x * uu).sum(dim=(-2, -1))  # [B, K]
        y = (P_x * vv).sum(dim=(-2, -1))  # [B, K]
        
        return torch.stack([x, y], dim=-1)  # [B, K, 2]
    
    def gaussian_mask(self, Psi_x: Tensor, H: int, W: int, sigma: float) -> Tensor:
        """
        Generate Gaussian masks centered at keypoints.

        Args:
            Psi_x: [B, K, 2]
        Returns:
            masks:  [B, K, H', W']
        """
        B, K, _ = Psi_x.shape
        device, dtype = Psi_x.device, Psi_x.dtype

        u = torch.arange(W, device=device, dtype=dtype)  # [W]
        v = torch.arange(H, device=device, dtype=dtype)  # [H]

        uu, vv = torch.meshgrid(v, u, indexing='xy')  # [W], [H] -> [H, W]

        uu = uu[None, None]  # [1, 1, H, W]
        vv = vv[None, None]  # [1, 1, H, W]

        du = uu - Psi_x[..., 0][..., None, None] # [B, K, H, W]
        dv = vv - Psi_x[..., 1][..., None, None] # [B, K, H, W]

        H = torch.exp(-(du**2 + dv**2) / (2 * sigma**2))  # [B, K, H, W]
        return H
    
    def transport(self, Phi_s: Tensor, H_s: Tensor, Phi_t: Tensor, H_t: Tensor) -> Tensor:
        """
        Transport keypoints from source to target using heatmap correlations.

        Args:
            Phi_s: [B, D, H', W'] source keypoints
            H_s:   [B, K, H', W'] source heatmaps
            Phi_t: [B, D, H', W'] target keypoints
            H_t:   [B, K, H', W'] target heatmaps
        Returns:
            Phi_hat: [B, D, H', W'] transported keypoints
        """
        # Check if input shapes are valid
        if Phi_s.shape != Phi_t.shape:
            raise ValueError(f"Phi_s and Phi_t must match, got {Phi_s.shape} vs {Phi_t.shape}")
        if H_s.shape != H_t.shape:
            raise ValueError(f"H_s and H_t must match, got {H_s.shape} vs {H_t.shape}")
        if Phi_s.dim() != 4 or H_s.dim() != 4:
            raise ValueError("Expected Phi_* as [B,D,H,W] and H_* as [B,K,H,W]")
        
        # Check both feature and heatmap spatial sizes match
        B, D, H, W = Phi_s.shape
        _, K, Hh, Wh = H_s.shape
        if (H, W) != (Hh, Wh):
            raise ValueError(f"Feature and heatmap spatial sizes must match, got {(H,W)} vs {(Hh,Wh)}")
        
        # Combine heatmaps into a single spatial mask per image: [B, 1, H, W]
        M_s = H_s.sum(dim=1, keepdim=True)  # [B, 1, H, W]
        M_t = H_t.sum(dim=1, keepdim=True)  # [B, 1, H, W]

        Phi_hat = (1.0 - M_s) * (1.0 - M_t) * Phi_s + M_t * Phi_t  # [B, D, H', W']
        return Phi_hat
        

    def decode(self, Phi_hat: Tensor) -> Tensor:
        """
        Decode keypoints back to image space (if decoder is defined).

        Args:
            Phi_hat: [B, K, 2]
        Returns:
            x_hat: [B, C, H, W]
        """
        return self.decoder(Phi_hat)

