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
        # standard projection for global attention:
        # (B, N, 3*d_out) -> (B, N, 3, n_heads, d_head) -> (3, B, n_heads, N, d_head)
        qkv = qkv(x).reshape(B, N, 3, n_heads, d_head).permute(2, 0, 3, 1, 4)
    else:
        # projection for mask-unit attention:
        # After Unroll, the tokens of one mask unit are NOT contiguous — they are
        # strided through the sequence with the mask-unit index as the INNER (fast)
        # axis. So split N as (n_tk_per_mu, n_mask_units) with n_mask_units innermost,
        # then move n_mask_units next to the batch dim so attention runs per unit.
        # (B, N, 3*d_out) -> (B, L, n_mu, 3, n_heads, d_head) -> (3, B, n_heads, n_mu, L, d_head)
        # Treating n_mask_units as the OUTER/contiguous axis instead makes a unit's
        # window span the whole image, i.e. global (non-local) attention.
        qkv = qkv(x).reshape(B, n_tk_per_mu, n_mask_units, 3, n_heads, d_head).permute(3, 0, 4, 2, 1, 5)

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

    # Split the pooled axis as (stride_q, L // stride_q) with stride_q OUTERMOST, then
    # max over it. This pools tokens that are (L // stride_q) apart in the unrolled
    # sequence — the strided grouping produced by Unroll and used by the residual
    # MaxPoolNd skip. Pooling the *contiguous* sub-group instead would max spatially
    # adjacent tokens and misalign the attention branch from the skip connection.
    shape[pool_dim] = stride_q
    shape.insert(pool_dim + 1, L // stride_q)

    return q.reshape(*shape).max(dim=pool_dim).values


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

    def forward(self, x: torch.Tensor):
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
            out = F.scaled_dot_product_attention(q, k, v)  # (B, n_heads, n_mu, L_q, d_head)

            # Restore the unrolled sequence order: flatten as (L_q, n_mask_units) with
            # the mask-unit index innermost, matching how tokens entered. (B, h, n_mu,
            # L_q, d) -> (B, L_q, n_mu, h, d) -> (B, L_q * n_mu, d_out)
            out = out.transpose(1, 3).reshape(B, -1, self.d_out)

        return self.proj(out)
