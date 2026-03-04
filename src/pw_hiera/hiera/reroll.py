import torch
import torch.nn as nn

from .types import Int2d


def undo_windowing(
    x: torch.Tensor,
    sz_tk: Int2d,
    sz_mu_tk: Int2d,
):
    """Undo mask-unit windowing and restore 2D spatial layout (B, H, W, C)."""

    # x is of shape (B, n_mu, mu_h, mu_w, C) where n_mu is number of mask-units in total.

    H, W = sz_tk

    # mask-unit shape in tokens
    mu_h, mu_w = sz_mu_tk

    if H % mu_h != 0 or W % mu_w != 0:
        raise ValueError("shape must be divisible by mu_shape")

    B, C = x.shape[0], x.shape[-1]

    # number of mask-units along each dim
    n_mu_h, n_mu_w = H // mu_h, W // mu_w

    # convert flat n_mu sequence into 2D grid of mask-units (n_mu -> n_mu_h, n_mu_w)
    x = x.view(B, n_mu_h, n_mu_w, mu_h, mu_w, C)

    # interleave grid position with within-unit position so that reshape merges them into the full spatial grid
    x = x.permute(0, 1, 3, 2, 4, 5)  # (B, n_mu_h, mu_h, n_mu_w, mu_w, C)
    return x.reshape(B, H, W, C)


class Unroll(nn.Module):
    """Reorder token sequence so q-pooling windows are contiguous."""

    # after patch embedding, we have a token sequence like A A B B A A B B C C D D C C D D. But to apply pooling and
    # attention, we need contiguous tokens, i.e. A A A A B B B B C C D D C C D D. This is what Unroll does. (Reroll is its
    # inverse, which restores original spatial layout.)

    def __init__(self, sz_tk: Int2d, unroll_schedule: list[Int2d]):
        super().__init__()
        self.size = sz_tk
        self.schedule = (
            unroll_schedule  # we need a schedule because we have multiple pool stages. corresponds to q-pool strides
        )

    def forward(self, x: torch.Tensor):
        B, N, C = x.shape
        H, W = self.size

        if N != H * W:
            raise ValueError("token count must match input token grid")

        mu_h, mu_w = 1, 1
        for sh, sw in self.schedule:
            mu_h *= sh
            mu_w *= sw

        if H % mu_h != 0 or W % mu_w != 0:
            raise ValueError("token grid must be divisible by final mask-unit shape")

        n_mu_h, n_mu_w = H // mu_h, W // mu_w

        # split flat token sequence into 2D grid of mask-units
        x = x.view(B, H, W, C)
        x = x.view(B, n_mu_h, mu_h, n_mu_w, mu_w, C)
        # group grid dims (n_mu_h, n_mu_w) and within-unit dims (mu_h, mu_w) adjacently
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()  # (B, n_mu_h, n_mu_w, mu_h, mu_w, C)
        x = x.view(
            B, n_mu_h * n_mu_w, mu_h, mu_w, C
        )  # flatten mask-unit grid back into sequence, but now each unit is contiguous

        # progressively split each mask-unit into sub-blocks (one level per pool stage, in reverse).
        # result: tokens at the same position within their mask-unit become adjacent,
        # so q-pooling over consecutive tokens reduces spatial resolution within each unit.
        for sh, sw in reversed(self.schedule):
            n_mu = x.shape[1]
            if mu_h % sh != 0 or mu_w % sw != 0:
                raise ValueError("mask-unit size must be divisible by stride")

            # split each mask-unit into sh*sw sub-blocks
            x = x.view(B, n_mu, sh, mu_h // sh, sw, mu_w // sw, C)
            # move sub-block dims (sh, sw) into the sequence dim
            x = x.permute(0, 2, 4, 1, 3, 5, 6).contiguous()

            mu_h //= sh
            mu_w //= sw

            # n_mu grows by sh*sw, mask-unit dims shrink accordingly
            x = x.view(B, n_mu * sh * sw, mu_h, mu_w, C)

        return x.view(B, -1, C)


class Reroll(nn.Module):
    """Inverse operation of Unroll for extracting intermediate 2D features."""

    # (reroll is much more complex than Unroll, because Unroll runs once after patch embed while reroll needs to
    # run after each stage end to restore 2D layout for intermediate feature extraction, thus needs to handle various schedules)

    def __init__(
        self,
        sz_in_px: Int2d,
        stride_patch_px: Int2d,
        unroll_schedule: list[Int2d],
        stage_ends: list[int],
        n_q_pool: int,
        sz_mask_unit_tk: Int2d,
    ):
        super().__init__()
        self.size = (sz_in_px[0] // stride_patch_px[0], sz_in_px[1] // stride_patch_px[1])
        self.schedule: dict[int, tuple[tuple[Int2d, ...], Int2d]] = {}
        self.masked_mu_shape_by_block: dict[int, Int2d] = {}
        size = self.size
        schedule = list(unroll_schedule)
        mu_h, mu_w = sz_mask_unit_tk
        pool_stage_ends = set(stage_ends[:n_q_pool])

        # precompute schedule for each stage block so we can efficiently reroll to 2D at each stage end. schedule is cumulative, i.e. by the last stage end we have the full unroll schedule, and in earlier stages we have only the relevant prefix of the schedule.
        for block_idx in range(stage_ends[-1] + 1):
            self.schedule[block_idx] = (tuple(schedule), size)
            self.masked_mu_shape_by_block[block_idx] = (mu_h, mu_w)

            if block_idx in pool_stage_ends and schedule:
                sh, sw = schedule[0]
                if mu_h % sh != 0 or mu_w % sw != 0:
                    raise ValueError("sz_mask_unit_tk must be divisible by stride_q_tk at each pool")
                size = (size[0] // sh, size[1] // sw)
                mu_h //= sh
                mu_w //= sw
                schedule = schedule[1:]

    def forward(
        self,
        x: torch.Tensor,
        block_idx: int,
        mask: torch.Tensor | None = None,
    ):
        schedule, size = self.schedule[block_idx]
        B, N, C = x.shape

        if mask is not None:
            mu_h, mu_w = self.masked_mu_shape_by_block[block_idx]
            n_tk_per_mu = mu_h * mu_w
            if N % n_tk_per_mu != 0:
                raise ValueError("masked token count must be divisible by mask-unit size")
            return x.view(B, -1, mu_h, mu_w, C)

        cur_mu_h, cur_mu_w = 1, 1
        for sh, sw in schedule:
            if N % (sh * sw) != 0:
                raise ValueError("token count must be divisible by stride product")

            x = x.view(B, sh, sw, N // (sh * sw), cur_mu_h, cur_mu_w, C)
            x = x.permute(0, 3, 1, 4, 2, 5, 6).contiguous()

            cur_mu_h *= sh
            cur_mu_w *= sw
            x = x.view(B, -1, cur_mu_h, cur_mu_w, C)
            N = x.shape[1]

        x = x.view(B, N, cur_mu_h, cur_mu_w, C)
        return undo_windowing(x, size, (cur_mu_h, cur_mu_w))
