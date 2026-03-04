import torch
import torch.nn as nn
import torch.nn.functional as F


def _project_qkv(
    x: torch.Tensor,
    qkv: nn.Linear,
    n_heads: int,
    d_head: int,
    *,
    n_mask_units: int | None = None,
    n_tk_per_mu: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Projects input x to Q, K, V tensors for attention and reshapes to separate heads."""

    # (n_heads * d_head * 3 is output dim of qkv projection)

    B, N, _ = x.shape
    if n_mask_units is None:
        # standard projection for global attention: (B, N, 3*d_out) -> (B, N, 3, n_heads, d_head) -> (3, B, n_heads, N, d_head)
        qkv = qkv(x).reshape(B, N, 3, n_heads, d_head).permute(2, 0, 3, 1, 4)
    else:
        # projection for mask-unit attention: (B, N, 3*d_out) -> (B, n_mu, L, 3, n_heads, d_head) -> (3, B, n_heads, n_mu, L, d_head)
        # instead of N (sequence length), we have n_mu (number of mask units) and L (tokens per mask unit) where n_mu is treated
        # as another batch dimension by pytorch. Thus, attention is computed separately per mask unit, and we can pool within it if needed
        qkv = qkv(x).reshape(B, n_mask_units, n_tk_per_mu, 3, n_heads, d_head).permute(3, 0, 4, 1, 2, 5)

    # unbind to return q, k, v separately
    return qkv.unbind(0)


def _pool_q(q: torch.Tensor, stride_q: int, *, pool_dim: int, error_message: str):
    """Applies max pooling to query tensor aloing specified dimension"""
    if stride_q <= 1:
        # no pooling, return as is
        return q

    L = q.shape[pool_dim]

    if L % stride_q != 0:
        # require divisibility, else we would have complex edge cases
        raise ValueError(error_message)

    shape = list(q.shape)

    # (B, n_heads, L, d_head) -> (B, n_heads, L//stride_q d_head)
    shape[pool_dim] = L // stride_q
    # insert new dim to correct for mismatch: (B, n_heads, L//stride_q, stride_q, d_head)
    shape.insert(pool_dim + 1, stride_q)

    # reshape our q and pool with max, thus collapsing into pooled query of shape (B, n_heads, L//stride_q, d_head)
    return q.reshape(*shape).max(dim=pool_dim + 1).values


class Attention(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        n_heads: int,
        *,
        stride_q_tk: int = 1,
        n_tk_per_mu: int | None = None,
    ):
        super().__init__()
        if d_out % n_heads != 0:
            raise ValueError("d_out must be divisible by n_heads")

        self.n_heads = n_heads
        self.d_head = d_out // n_heads
        self.d_out = d_out
        self.stride_q_tk = stride_q_tk
        self.n_tk_per_mu = n_tk_per_mu

        self.qkv = nn.Linear(d_in, 3 * d_out)
        self.proj = nn.Linear(d_out, d_out)

    def forward(self, x: torch.Tensor) :
        B, N, _ = x.shape  # (B, N, d_in)

        if self.n_tk_per_mu is None:
            # simple global attention case
            q, k, v = _project_qkv(x, self.qkv, self.n_heads, self.d_head)

            # apply q-pooling
            q = _pool_q(q, self.stride_q_tk, pool_dim=2, error_message="seq_len must be divisible by stride_q_tk")

            # apply attention
            out = F.scaled_dot_product_attention(q, k, v)  # (B, h, L, d)

            # reshape back to (B, N, d_out)
            out = out.transpose(1, 2).reshape(B, q.shape[2], self.d_out)
        else:
            if N % self.n_tk_per_mu != 0:
                raise ValueError("sequence length must be divisible by n_tk_per_mu")

            n_mask_units = N // self.n_tk_per_mu

            q, k, v = _project_qkv(
                x,
                self.qkv,
                self.n_heads,
                self.d_head,
                n_mask_units=n_mask_units,
                n_tk_per_mu=self.n_tk_per_mu,
            )

            # apply q-pooling WITHIN mask-units
            q = _pool_q(q, self.stride_q_tk, pool_dim=3, error_message="win_len must divide by stride_q_tk")
            out = F.scaled_dot_product_attention(q, k, v)  # (B, n_heads, n_mu, L, d_head)

            # reshape back to (B, N, d_out)
            out = out.permute(0, 2, 3, 1, 4).reshape(B, n_mask_units * q.shape[3], self.d_out)

        return self.proj(out)
