import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from hiera_2d.hiera.mae import DecoderBlock
from hiera_2d.hiera.model import Hiera
from hiera_2d.hiera.types import Model


class ARHeadConfig(Model):
    d_head: int = 256
    n_blocks: int = 4
    n_heads: int = 8
    mlp_ratio: float = 4.0
    predict_residual: bool = False


class HieraAR(nn.Module):
    """Autoregressive next-frame predictor built on a Hiera encoder.

    Encoder is initialized from MAE pretrained weights and fine-tuned end-to-end.
    Predicts next frame pixels from current frame pixels.
    """

    def __init__(self, encoder: Hiera, config: ARHeadConfig):
        super().__init__()
        self.encoder = encoder
        self.config = config
        self.shapes = encoder.shapes

        d_enc = encoder.d_encoder_out
        n_tokens = math.prod(self.shapes.sz_tk_final)

        self.encoder_norm = nn.LayerNorm(d_enc)
        self.proj_in = nn.Linear(d_enc, config.d_head)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens, config.d_head))

        self.blocks = nn.ModuleList(
            [DecoderBlock(config.d_head, config.n_heads, config.mlp_ratio) for _ in range(config.n_blocks)]
        )

        self.norm = nn.LayerNorm(config.d_head)

        stride = encoder.config.patch_embed.stride_px[0] * (encoder.config.stride_q_tk[0] ** self.shapes.n_q_pool)
        self.stride_pred_px = stride
        self.pred = nn.Linear(config.d_head, stride**2 * encoder.config.n_channels)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for name, m in self.named_modules():
            # only init AR head layers, not the pretrained encoder
            if name.startswith("encoder"):
                continue
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a frame. Returns flattened spatial tokens."""
        _, intermediates = self.encoder.forward_with_intermediates(x, mask=None)
        feat = intermediates[-1]  # (B, H', W', d_enc)
        B, H, W, D = feat.shape
        return feat.reshape(B, H * W, D)

    def predict(self, tokens: torch.Tensor) -> torch.Tensor:
        """Predict next-frame pixels from encoder tokens."""
        x = self.encoder_norm(tokens)
        x = self.proj_in(x)
        x = x + self.pos_embed

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        x = self.pred(x)  # (B, N, stride^2 * C)
        return self._to_image(x)

    def _to_image(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape token predictions to (B, C, H, W)."""
        B = x.shape[0]
        C = self.encoder.config.n_channels
        ps = self.stride_pred_px
        H_tk, W_tk = self.shapes.sz_tk_final

        x = x.reshape(B, H_tk, W_tk, ps, ps, C)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        return x.reshape(B, C, H_tk * ps, W_tk * ps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict next frame from current frame.

        With predict_residual the head predicts the change x_{t+1} - x_t and we
        add it back to the input. At small dt this removes the trivial
        "copy the input" solution and concentrates the signal on the dynamics.

        Args:
            x: (B, C, H, W) current frame
        Returns:
            (B, C, H, W) predicted next frame
        """
        out = self.predict(self.encode(x))
        return x + out if self.config.predict_residual else out

    @torch.no_grad()
    def rollout(self, x: torch.Tensor, n_steps: int) -> torch.Tensor:
        """Autoregressively roll out n_steps into the future.

        Args:
            x: (B, C, H, W) initial frame
            n_steps: number of future frames to predict
        Returns:
            (B, n_steps, C, H, W) predicted trajectory
        """
        preds = []
        for _ in range(n_steps):
            x = self.forward(x)
            preds.append(x)
        return torch.stack(preds, dim=1)


def ar_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)
