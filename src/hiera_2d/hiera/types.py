import math

from pydantic import BaseModel, ConfigDict

type Int2d = tuple[int, int]


class Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class HieraShapes:
    """Derived spatial shapes and token counts for Hiera (px/tk/mu units)."""

    def __init__(
        self,
        *,
        sz_in_px: Int2d,
        stride_patch_px: Int2d,
        sz_mask_unit_tk: Int2d,
        stride_q_tk: Int2d,
        n_q_pool: int,
    ):
        if n_q_pool < 0:
            raise ValueError("n_q_pool must be >= 0")
        if len(sz_in_px) != 2 or len(stride_patch_px) != 2 or len(sz_mask_unit_tk) != 2 or len(stride_q_tk) != 2:
            raise ValueError("HieraShapes currently supports 2D shapes only")
        if any(v <= 0 for v in (*sz_in_px, *stride_patch_px, *sz_mask_unit_tk, *stride_q_tk)):
            raise ValueError("all spatial sizes and strides must be > 0")
        if any(i % s != 0 for i, s in zip(sz_in_px, stride_patch_px, strict=True)):
            raise ValueError("sz_in_px must be divisible by stride_patch_px")

        sz_tk = tuple(i // s for i, s in zip(sz_in_px, stride_patch_px, strict=True))

        if any(t % m != 0 for t, m in zip(sz_tk, sz_mask_unit_tk, strict=True)):
            raise ValueError("sz_tk must be divisible by sz_mask_unit_tk")

        pool_factors = tuple(s**n_q_pool for s in stride_q_tk)
        if any(t % p != 0 for t, p in zip(sz_tk, pool_factors, strict=True)):
            raise ValueError("sz_tk must be divisible by stride_q_tk ** n_q_pool")
        if any(m % p != 0 for m, p in zip(sz_mask_unit_tk, pool_factors, strict=True)):
            raise ValueError("sz_mask_unit_tk must be divisible by stride_q_tk ** n_q_pool")

        self.sz_in_px = sz_in_px
        self.stride_patch_px = stride_patch_px
        self.sz_mask_unit_tk = sz_mask_unit_tk
        self.stride_q_tk = stride_q_tk
        self.n_q_pool = n_q_pool
        self.sz_tk = sz_tk
        self.n_tokens = math.prod(sz_tk)
        self.sz_mu = tuple(t // m for t, m in zip(sz_tk, sz_mask_unit_tk, strict=True))
        self.sz_tk_final = tuple(t // p for t, p in zip(sz_tk, pool_factors, strict=True))
        self.sz_mask_unit_tk_final = tuple(m // p for m, p in zip(sz_mask_unit_tk, pool_factors, strict=True))

    @property
    def n_mu(self):
        return math.prod(self.sz_mu)

    @property
    def n_tk_per_mu(self):
        return math.prod(self.sz_mask_unit_tk)

    @property
    def stride_q_flat(self):
        return math.prod(self.stride_q_tk)
