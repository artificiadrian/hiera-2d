"""Spectral diagnostics for 2D velocity fields.

The central plot for the thesis is the radially-averaged kinetic energy
spectrum E(k): ground truth is the reference (dark) line, and model rollouts
are overlaid. A model with strong spectral bias matches low wavenumbers but
loses energy at high k; we want pretraining to keep the high-k tail close to
ground truth.
"""

import numpy as np


def _radial_wavenumber(n: int) -> np.ndarray:
    """Magnitude |k| of the integer wavenumber at each FFT grid point (n, n)."""
    k = np.fft.fftfreq(n, d=1.0 / n)  # integer wavenumbers: 0, 1, ..., -1
    kx, ky = np.meshgrid(k, k, indexing="ij")
    return np.sqrt(kx**2 + ky**2)


def radial_energy_spectrum(velocity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Radially-averaged kinetic energy spectrum of 2D velocity fields.

    Args:
        velocity: (..., 2, H, W) array, channels are (u, v). H must equal W.

    Returns:
        k: (K,) integer wavenumbers, 0 .. H//2.
        spectrum: (..., K) energy summed over each wavenumber shell.
    """
    velocity = np.asarray(velocity, dtype=np.float64)
    *lead, two, h, w = velocity.shape
    if two != 2:
        raise ValueError(f"expected 2 velocity channels, got {two}")
    if h != w:
        raise ValueError(f"expected square fields, got {h}x{w}")
    n = h

    kbin = np.round(_radial_wavenumber(n)).astype(int).ravel()
    n_bins = n // 2 + 1

    flat = velocity.reshape(-1, 2, n, n)
    out = np.empty((flat.shape[0], n_bins))
    for i, field in enumerate(flat):
        uh = np.fft.fft2(field[0]) / (n * n)
        vh = np.fft.fft2(field[1]) / (n * n)
        energy = 0.5 * (np.abs(uh) ** 2 + np.abs(vh) ** 2)
        out[i] = np.bincount(kbin, weights=energy.ravel(), minlength=n_bins)[:n_bins]

    k = np.arange(n_bins)
    return k, out.reshape(*lead, n_bins)


def power_db(spectrum: np.ndarray, eps: float = 1e-20) -> np.ndarray:
    """Convert an energy spectrum to decibels: 10*log10(E)."""
    return 10.0 * np.log10(np.asarray(spectrum) + eps)


def vorticity(velocity: np.ndarray, domain_extent: float = 2.0 * np.pi) -> np.ndarray:
    """Vorticity w = dv/dx - du/dy via spectral differentiation.

    Nicer to look at than raw velocity components for Kolmogorov flow.

    Args:
        velocity: (..., 2, H, W) array, channels (u, v).
    Returns:
        (..., H, W) vorticity field.
    """
    velocity = np.asarray(velocity, dtype=np.float64)
    *lead, two, h, w = velocity.shape
    if two != 2 or h != w:
        raise ValueError(f"expected (..., 2, N, N), got {velocity.shape}")
    n = h

    k = np.fft.fftfreq(n, d=domain_extent / n) * 2.0 * np.pi
    kx, ky = np.meshgrid(k, k, indexing="ij")

    flat = velocity.reshape(-1, 2, n, n)
    out = np.empty((flat.shape[0], n, n))
    for i, field in enumerate(flat):
        uh = np.fft.fft2(field[0])
        vh = np.fft.fft2(field[1])
        out[i] = np.real(np.fft.ifft2(1j * kx * vh - 1j * ky * uh))
    return out.reshape(*lead, n, n)
