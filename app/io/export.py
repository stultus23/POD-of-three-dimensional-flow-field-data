"""Export utilities."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from app.utils.models import ExportSettings, PodResult, to_jsonable


def _sanitize_filename(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_") or "var"


def export_results(result: PodResult, settings: ExportSettings, out_dir: str) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    metadata = {
        "snapshot_shape": result.snapshot_shape,
        "grid_shape": result.grid_shape,
        "variables": result.variables,
        "n_modes": int(result.modes.shape[1]),
    }
    (out_path / "metadata.json").write_text(json.dumps(to_jsonable(metadata), indent=2), encoding="utf-8")

    if settings.export_eigs:
        eigs = (result.singular_values ** 2) / (result.snapshot_shape[1] - 1)

    if settings.export_modes and settings.export_npy:
        np.savez(
            out_path / "modes.npz",
            modes=result.modes,
            mean=result.mean,
            coords=result.coords,
        )

    if settings.export_csv:
        coords = result.coords
        if coords is not None:
            n_points = coords.shape[0]
            n_vars = len(result.variables)
        else:
            n_points = 0
            n_vars = 0
        grid_shape = result.grid_shape
        if coords is not None and grid_shape is None:
            try:
                xs = np.unique(coords[:, 0])
                ys = np.unique(coords[:, 1])
                zs = np.unique(coords[:, 2])
                if xs.size * ys.size * zs.size == coords.shape[0]:
                    grid_shape = (int(xs.size), int(ys.size), int(zs.size))
            except Exception:
                grid_shape = None

        if settings.export_modes:
            np.savetxt(out_path / "mean.csv", result.mean, delimiter=",")
            if coords is not None:
                np.savetxt(out_path / "coords.csv", coords, delimiter=",")

        # Metrics summary (one column per metric, one row per mode)
        if settings.export_metrics and settings.export_eigs:
            wide_path = out_path / "modal_metrics_wide.csv"
            with wide_path.open("w", encoding="utf-8") as f:
                f.write("Mode,Energy,CumulativeEnergy,Eigenvalue,Amplitude\n")
                for i in range(result.modes.shape[1]):
                    row = [
                        f"Mode{i+1:03d}",
                        str(result.energy[i]),
                        str(result.cumulative_energy[i]),
                        str(eigs[i]),
                        str(result.singular_values[i]),
                    ]
                    f.write(",".join(row) + "\n")

        if settings.export_modes or settings.export_coeffs:
            per_mode_dir = out_path / "per_mode"
            per_mode_dir.mkdir(exist_ok=True)

            for i in range(result.modes.shape[1]):
                mode_dir = per_mode_dir / f"mode_{i+1:03d}"
                mode_dir.mkdir(exist_ok=True)
                if settings.export_modes:
                    np.savetxt(mode_dir / "mode.csv", result.modes[:, i], delimiter=",")
                if settings.export_coeffs:
                    coeff = result.coeffs[i, :]
                    np.savetxt(mode_dir / "coefficients.csv", coeff, delimiter=",")
                    if settings.export_fft:
                        # FFT; use sample frequency if provided, else unit spacing
                        if settings.sample_freq_hz and settings.sample_freq_hz > 0:
                            d = 1.0 / settings.sample_freq_hz
                        else:
                            d = 1.0
                        coeff_d = coeff.astype(float) - coeff.mean()
                        freq = np.fft.rfftfreq(coeff_d.size, d=d)
                        fft = np.fft.rfft(coeff_d)
                        amp = np.abs(fft) / coeff_d.size * 2.0
                        if amp.size > 0:
                            amp[0] = amp[0] / 2.0
                            if coeff_d.size % 2 == 0:
                                amp[-1] = amp[-1] / 2.0
                        fft_out = np.column_stack((freq, amp))
                        np.savetxt(mode_dir / "coefficients_fft.csv", fft_out, delimiter=",", header="freq,amplitude", comments="")

        if settings.export_cloud:
            if coords is not None and n_points > 0 and result.modes.shape[0] == n_points * n_vars and n_vars >= 3:
                clouds_dir = out_path / "clouds"
                clouds_dir.mkdir(exist_ok=True)
                for i in range(result.modes.shape[1]):
                    dat_path = clouds_dir / f"mode_{i+1:03d}.dat"
                    mode_fields = result.modes[:, i].reshape(n_points, n_vars)
                    u = mode_fields[:, 0]
                    v = mode_fields[:, 1]
                    w = mode_fields[:, 2]
                    length = np.sqrt(u * u + v * v + w * w)
                    status = np.full((n_points, 1), 16.0)
                    data = np.column_stack((coords, status, u, v, w, length))
                    with dat_path.open("w", encoding="utf-8") as f:
                        f.write('TITLE="DynamicStudio Exported Data"\n')
                        f.write('VARIABLES= "X (m)[m]" "Y (m)[m]" "Z (m)[m]" "Status[-]" "U[-]" "V[-]" "W[-]" "Length[-]"\n')
                        if grid_shape is not None:
                            i_dim, j_dim, k_dim = grid_shape
                            zone = f'ZONE T="DynamicStudio Data" I={i_dim} J={j_dim} K={k_dim} F=POINT'
                        else:
                            zone = 'ZONE T="DynamicStudio Data" F=POINT'
                        f.write(zone + "\n")
                        np.savetxt(f, data, fmt="%.15g", delimiter=" ")

    if settings.export_recon and settings.export_csv:
        recon_dir = out_path / "recon"
        recon_dir.mkdir(exist_ok=True)
        k = max(1, min(settings.recon_modes, result.modes.shape[1]))
        recon = result.mean[:, None] + result.modes[:, :k] @ result.coeffs[:k, :]
        # Export reconstructed fields as DAT snapshots if coords are available
        coords = result.coords
        if coords is not None:
            n_points = coords.shape[0]
            n_vars = len(result.variables)
            if result.modes.shape[0] == n_points * n_vars:
                recon_fields = recon.reshape(n_points, n_vars, -1)
                for t in range(recon_fields.shape[2]):
                    dat_path = recon_dir / f"recon_{t:03d}.dat"
                    u = recon_fields[:, 0, t]
                    v = recon_fields[:, 1, t] if n_vars > 1 else np.zeros_like(u)
                    w = recon_fields[:, 2, t] if n_vars > 2 else np.zeros_like(u)
                    length = np.sqrt(u * u + v * v + w * w)
                    status = np.full((n_points, 1), 16.0)
                    data = np.column_stack((coords, status, u, v, w, length))
                    with dat_path.open("w", encoding="utf-8") as f:
                        f.write('TITLE="DynamicStudio Exported Data"\n')
                        f.write('VARIABLES= "X (m)[m]" "Y (m)[m]" "Z (m)[m]" "Status[-]" "U[-]" "V[-]" "W[-]" "Length[-]"\n')
                        if result.grid_shape is not None:
                            i_dim, j_dim, k_dim = result.grid_shape
                            zone = f'ZONE T="DynamicStudio Data" I={i_dim} J={j_dim} K={k_dim} F=POINT'
                        else:
                            zone = 'ZONE T="DynamicStudio Data" F=POINT'
                        f.write(zone + "\n")
                        np.savetxt(f, data, fmt="%.15g", delimiter=" ")
