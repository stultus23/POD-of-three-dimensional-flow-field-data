"""Minimal Tkinter UI to avoid heavy GUI dependencies."""

from __future__ import annotations

import threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from app.io.dat_parser import parse_dat_file, preview_table, inspect_header
from app.io.export import export_results
from app.pod.pod_compute import compute_pod
from app.utils.models import ExportSettings, ParseSettings, PodSettings
from app.utils.validators import validate_snapshots


def run_tk_ui() -> None:
    root = tk.Tk()
    root.title("POD Flow")
    root.geometry("1400x800")

    files: list[str] = []
    state = {"snapshots": None, "result": None, "recon_ready": False}

    def add_files() -> None:
        nonlocal files
        selected = filedialog.askopenfilenames(filetypes=[("DAT files", "*.dat *.DAT"), ("All files", "*.*")])
        if selected:
            files.extend(selected)
            listbox.delete(0, tk.END)
            for f in files:
                listbox.insert(tk.END, f)

    def remove_selected() -> None:
        nonlocal files
        sel = list(listbox.curselection())
        if not sel:
            return
        for idx in reversed(sel):
            files.pop(idx)
        listbox.delete(0, tk.END)
        for f in files:
            listbox.insert(tk.END, f)

    def choose_out() -> None:
        out = filedialog.askdirectory()
        if out:
            out_var.set(out)

    def log(msg: str) -> None:
        log_text.configure(state="normal")
        log_text.insert(tk.END, msg + "\n")
        log_text.configure(state="disabled")
        log_text.see(tk.END)

    def _parse_settings() -> ParseSettings:
        delim = delim_var.get()
        parse = ParseSettings(
            delimiter=delim if delim else " ",
            header_rows=int(header_var.get()),
            has_coords=coords_var.get(),
        )
        data_vars_text = data_vars_var.get().strip()
        if data_vars_text:
            parse.data_vars = tuple(x.strip() for x in data_vars_text.split(",") if x.strip())
        return parse

    def _pod_settings() -> PodSettings:
        n_modes = int(modes_var.get())
        return PodSettings(
            mean_subtract=mean_var.get(),
            normalize=norm_var.get(),
            n_modes=None if n_modes == 0 else n_modes,
            method=method_var.get(),
        )

    def detect_header() -> None:
        if not files:
            messagebox.showwarning("No files", "Please add DAT files first.")
            return
        info = inspect_header(files[0])
        vars_list = info.get("variables", [])
        grid = info.get("grid_shape")
        vars_label.config(text=f"Variables: {', '.join(vars_list) if vars_list else '-'}")
        grid_label.config(text=f"Grid: {grid if grid else '-'}")
        # auto-fill common UVW
        if vars_list and all(v in vars_list for v in ["U[m/s]", "V[m/s]", "W[m/s]"]):
            data_vars_var.set("U[m/s],V[m/s],W[m/s]")
        # refresh preview so the table stays visible after detection
        try:
            preview_data()
        except Exception:
            pass

    def preview_data() -> None:
        if not files:
            messagebox.showwarning("No files", "Please add DAT files first.")
            return
        try:
            settings = _parse_settings()
            snap = parse_dat_file(files[0], settings)
            data = np.column_stack((snap.coords, snap.data)) if snap.coords is not None else snap.data
            data = data[:10, :]
            # Pop preview in a separate window
            win = tk.Toplevel(root)
            win.title("Data Preview")
            win.geometry("800x300")
            table = ttk.Treeview(win, show="headings")
            table.pack(fill=tk.BOTH, expand=True)
            table["columns"] = [str(i) for i in range(data.shape[1])]
            for c in table["columns"]:
                table.heading(c, text=c)
                table.column(c, width=80)
            for row in data:
                table.insert("", tk.END, values=[f"{v:.6g}" for v in row])
        except Exception as exc:
            messagebox.showerror("Preview error", str(exc))

    def update_plots() -> None:
        result_cache = state["result"]
        if result_cache is None:
            return
        energy_ax.clear()
        cum_ax.clear()
        energy_ax.plot(np.arange(1, len(result_cache.energy) + 1), result_cache.energy, marker="o")
        cum_ax.plot(np.arange(1, len(result_cache.cumulative_energy) + 1), result_cache.cumulative_energy, marker="o")
        energy_ax.set_title("Energy Spectrum")
        cum_ax.set_title("Cumulative Energy")
        fft_ax.clear()
        cloud_ax.clear()
        mode_idx = cloud_mode_var.get()
        if result_cache.coords is not None and 1 <= mode_idx <= result_cache.modes.shape[1]:
            coords = result_cache.coords
            n_points = coords.shape[0]
            n_vars = len(result_cache.variables)
            if result_cache.modes.shape[0] == n_points * n_vars:
                z_vals = coords[:, 2]
                unique_z = np.unique(z_vals)
                z0 = None
                if unique_z.size > 0:
                    sel = z_slice_var.get()
                    if sel == "Auto":
                        z0 = unique_z[np.argmin(np.abs(unique_z - 0.0))]
                    else:
                        try:
                            z0 = float(sel)
                        except ValueError:
                            z0 = unique_z[0]
                    mask = z_vals == z0
                else:
                    mask = slice(None)
                mode_fields = result_cache.modes[:, mode_idx - 1].reshape(n_points, n_vars)
                var_name = cloud_var_combo.get()
                if var_name == "Length":
                    if n_vars >= 3:
                        u = mode_fields[:, 0]
                        v = mode_fields[:, 1]
                        w = mode_fields[:, 2]
                        values = np.sqrt(u * u + v * v + w * w)
                    else:
                        values = mode_fields[:, 0]
                else:
                    try:
                        vidx = result_cache.variables.index(var_name)
                        values = mode_fields[:, vidx]
                    except ValueError:
                        values = mode_fields[:, 0]
                x = coords[mask, 0]
                y = coords[mask, 1]
                v = values[mask]
                # Use a smooth colormap via tricontourf for a filled cloud-like plot
                cloud_ax.tricontourf(x, y, v, levels=60, cmap="viridis")
                cloud_ax.set_aspect("equal", adjustable="box")
                z_text = f"Z={z0:.3f}" if z0 is not None else "Z=?"
                cloud_ax.set_title(f"Mode {mode_idx:03d} - {var_name} ({z_text})")
        # FFT plot (selectable mode)
        fft_mode_idx = fft_mode_var.get()
        if 1 <= fft_mode_idx <= result_cache.coeffs.shape[0]:
            coeff = result_cache.coeffs[fft_mode_idx - 1, :].astype(float)
            coeff = coeff - coeff.mean()
            sf_text = sample_freq_var.get().strip()
            if sf_text:
                try:
                    fs = float(sf_text)
                    d = 1.0 / fs if fs > 0 else 1.0
                except ValueError:
                    d = 1.0
            else:
                d = 1.0
            freq = np.fft.rfftfreq(coeff.size, d=d)
            fft = np.fft.rfft(coeff)
            amp = np.abs(fft) / coeff.size * 2.0
            if amp.size > 0:
                amp[0] = amp[0] / 2.0
                if coeff.size % 2 == 0:
                    amp[-1] = amp[-1] / 2.0
            fft_ax.plot(freq, amp, color="#ff7f0e")
            fft_ax.set_title(f"FFT Mode {fft_mode_idx:03d}")
        energy_canvas.draw_idle()

    def run_pod() -> None:
        if not files:
            messagebox.showwarning("No files", "Please add DAT files first.")
            return

        def task() -> None:
            try:
                parse = _parse_settings()
                pod = _pod_settings()
                snaps = [parse_dat_file(p, parse) for p in files]
                ok, msg = validate_snapshots(snaps)
                if not ok:
                    raise ValueError(msg)
                result = compute_pod(snaps, pod)
                state["snapshots"] = snaps
                state["result"] = result
                log("POD completed")
                cloud_mode_var.set(1)
                modes_list = [f"Mode {i:03d}" for i in range(1, result.modes.shape[1] + 1)]
                cloud_mode_combo["values"] = modes_list
                cloud_mode_combo.set(modes_list[0] if modes_list else "")
                fft_mode_var.set(1)
                fft_mode_combo["values"] = modes_list
                fft_mode_combo.set(modes_list[0] if modes_list else "")
                var_list = list(result.variables)
                if "Length" not in var_list:
                    var_list.append("Length")
                cloud_var_combo["values"] = var_list
                cloud_var_combo.set(var_list[0] if var_list else "")
                # Populate available Z slices
                z_vals = np.unique(result.coords[:, 2]) if result.coords is not None else []
                z_options = ["Auto"] + [f"{z:.3f}" for z in z_vals]
                z_slice_combo["values"] = z_options
                z_slice_combo.set(z_options[0] if z_options else "Auto")
                export_btn.config(state="normal")
                state["recon_ready"] = False
                recon_check.config(state="normal")
                recon_export_check.config(state="disabled")
                update_plots()
                messagebox.showinfo("Done", "POD completed. You can export if needed.")
            except Exception as exc:
                log(f"Error: {exc}")
                messagebox.showerror("Error", str(exc))
            finally:
                progress.stop()

        progress.start(10)
        threading.Thread(target=task, daemon=True).start()

    def export_now() -> None:
        result_cache = state["result"]
        if result_cache is None:
            messagebox.showwarning("No results", "Run POD first.")
            return
        out_dir = out_var.get().strip()
        if not out_dir:
            messagebox.showwarning("No output", "Please choose output folder.")
            return
        try:
            sf_text = sample_freq_var.get().strip()
            sample_freq = float(sf_text) if sf_text else None
            if export_recon_var.get() and not state["recon_ready"]:
                messagebox.showwarning("Reconstruction", "Please reconstruct first.")
                return
            settings = ExportSettings(
                export_cloud=export_cloud_var.get(),
                export_metrics=export_metrics_var.get(),
                export_modes=export_modes_var.get(),
                export_coeffs=export_coeffs_var.get(),
                export_fft=export_fft_var.get(),
                sample_freq_hz=sample_freq,
                export_recon=export_recon_var.get(),
                recon_modes=int(recon_modes_var.get()) if recon_modes_var.get().strip() else 5,
            )
            export_results(result_cache, settings, out_dir)
            log("Export completed")
            messagebox.showinfo("Done", "Export completed.")
        except Exception as exc:
            log(f"Error: {exc}")
            messagebox.showerror("Error", str(exc))

    # Layout
    paned = ttk.Panedwindow(root, orient=tk.HORIZONTAL)
    paned.pack(fill=tk.BOTH, expand=True)

    left = ttk.Frame(paned, padding=10)

    listbox = tk.Listbox(left, width=48, height=10)
    listbox.pack(fill=tk.X)

    btn_row = ttk.Frame(left)
    btn_row.pack(fill=tk.X, pady=5)
    ttk.Button(btn_row, text="Add DAT", command=add_files).pack(side=tk.LEFT, padx=2)
    ttk.Button(btn_row, text="Remove", command=remove_selected).pack(side=tk.LEFT, padx=2)
    ttk.Label(btn_row, text="Sample Freq (Hz)").pack(side=tk.LEFT, padx=6)
    sample_freq_var = tk.StringVar(value="")
    ttk.Entry(btn_row, textvariable=sample_freq_var, width=10).pack(side=tk.LEFT)

    row_delim = ttk.Frame(left)
    row_delim.pack(fill=tk.X)
    ttk.Label(row_delim, text="Delimiter").pack(side=tk.LEFT)
    delim_var = tk.StringVar(value=" ")
    ttk.Entry(row_delim, textvariable=delim_var, width=8).pack(side=tk.LEFT, padx=4)
    ttk.Label(row_delim, text="Header rows").pack(side=tk.LEFT)
    header_var = tk.StringVar(value="3")
    ttk.Entry(row_delim, textvariable=header_var, width=6).pack(side=tk.LEFT, padx=4)

    coords_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(left, text="Has coordinates", variable=coords_var).pack(anchor=tk.W)

    ttk.Label(left, text="Data variables (comma-separated)").pack(anchor=tk.W)
    data_vars_var = tk.StringVar(value="")
    ttk.Entry(left, textvariable=data_vars_var).pack(fill=tk.X)

    detect_row = ttk.Frame(left)
    detect_row.pack(fill=tk.X, pady=4)
    ttk.Button(detect_row, text="Detect Header", command=detect_header).pack(side=tk.LEFT, padx=2)
    ttk.Button(detect_row, text="Preview", command=preview_data).pack(side=tk.LEFT, padx=2)

    vars_label = ttk.Label(left, text="Variables: -")
    vars_label.pack(anchor=tk.W)
    grid_label = ttk.Label(left, text="Grid: -")
    grid_label.pack(anchor=tk.W)

    row_mean = ttk.Frame(left)
    row_mean.pack(fill=tk.X, pady=2)
    mean_var = tk.BooleanVar(value=True)
    norm_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(row_mean, text="Subtract mean", variable=mean_var).pack(side=tk.LEFT)
    ttk.Checkbutton(row_mean, text="Normalize", variable=norm_var).pack(side=tk.LEFT, padx=6)

    row_modes = ttk.Frame(left)
    row_modes.pack(fill=tk.X, pady=2)
    ttk.Label(row_modes, text="Modes (0=auto)").pack(side=tk.LEFT)
    modes_var = tk.StringVar(value="0")
    ttk.Entry(row_modes, textvariable=modes_var, width=8).pack(side=tk.LEFT, padx=4)
    ttk.Label(row_modes, text="Method").pack(side=tk.LEFT)
    method_var = tk.StringVar(value="svd")
    ttk.Combobox(row_modes, textvariable=method_var, values=["svd", "randomized"], state="readonly", width=12).pack(side=tk.LEFT, padx=4)

    def reconstruct_now() -> None:
        if state["result"] is None:
            messagebox.showwarning("No results", "Run POD first.")
            return
        try:
            int(recon_modes_var.get())
        except ValueError:
            messagebox.showwarning("Invalid", "Recon modes must be an integer.")
            return
        state["recon_ready"] = True
        recon_export_check.config(state="normal")
        export_recon_var.set(True)
        log("Reconstruction ready (will be exported on demand).")

    recon_actions = ttk.LabelFrame(left, text="Run / Reconstruction")
    recon_actions.pack(fill=tk.X, pady=4)

    btn_row2 = ttk.Frame(recon_actions)
    btn_row2.pack(fill=tk.X, pady=2)
    btn_row2.columnconfigure(0, weight=1)
    btn_row2.columnconfigure(1, weight=1)

    run_btn = ttk.Button(btn_row2, text="Run POD", command=run_pod)
    run_btn.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=2, pady=2)
    run_btn.configure(width=10)
    run_btn.configure(padding=(10, 10))

    recon_modes_var = tk.StringVar(value="5")
    recon_modes_frame = ttk.Frame(btn_row2)
    recon_modes_frame.grid(row=0, column=1, sticky="nsew", padx=2, pady=2)
    ttk.Label(recon_modes_frame, text="Recon modes").pack(side=tk.LEFT)
    ttk.Entry(recon_modes_frame, textvariable=recon_modes_var, width=6).pack(side=tk.LEFT, padx=4)

    recon_check = ttk.Button(btn_row2, text="Reconstruct now", command=reconstruct_now, state="disabled")
    recon_check.grid(row=1, column=1, sticky="nsew", padx=2, pady=2)

    export_frame = ttk.LabelFrame(left, text="Export Options")
    export_frame.pack(fill=tk.X, pady=4)
    export_cloud_var = tk.BooleanVar(value=True)
    export_metrics_var = tk.BooleanVar(value=True)
    export_modes_var = tk.BooleanVar(value=False)
    export_coeffs_var = tk.BooleanVar(value=False)
    export_fft_var = tk.BooleanVar(value=True)
    export_recon_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(export_frame, text="Cloud DATs", variable=export_cloud_var).pack(anchor=tk.W)
    ttk.Checkbutton(export_frame, text="Energy/Cumulative/Eigenvalue/Amplitude", variable=export_metrics_var).pack(anchor=tk.W)
    ttk.Checkbutton(export_frame, text="Spatial modes (mode.csv)", variable=export_modes_var).pack(anchor=tk.W)
    ttk.Checkbutton(export_frame, text="Chronos (coefficients.csv)", variable=export_coeffs_var).pack(anchor=tk.W)
    ttk.Checkbutton(export_frame, text="FFT of Chronos (coefficients_fft.csv)", variable=export_fft_var).pack(anchor=tk.W)
    recon_export_check = ttk.Checkbutton(export_frame, text="Export reconstruction", variable=export_recon_var, state="disabled")
    recon_export_check.pack(anchor=tk.W)

    out_row = ttk.Frame(left)
    out_row.pack(fill=tk.X, pady=4)
    out_var = tk.StringVar(value="")
    ttk.Entry(out_row, textvariable=out_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
    ttk.Button(out_row, text="Browse", command=choose_out).pack(side=tk.LEFT, padx=2)

    export_btn = ttk.Button(left, text="Export Results", command=export_now, state="disabled")
    export_btn.pack(fill=tk.X, pady=4)

    progress = ttk.Progressbar(left, mode="indeterminate")
    progress.pack(fill=tk.X)

    right = ttk.Frame(paned, padding=10)
    paned.add(left, weight=1)
    paned.add(right, weight=4)

    # Plots
    plot_frame = ttk.LabelFrame(right, text="Results")
    plot_frame.pack(fill=tk.BOTH, expand=True, pady=6)
    fig = Figure(figsize=(10, 6), tight_layout=True)
    energy_ax = fig.add_subplot(221)
    cum_ax = fig.add_subplot(222)
    fft_ax = fig.add_subplot(223)
    cloud_ax = fig.add_subplot(224)
    energy_canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    energy_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # Cloud/FFT selectors
    mode_select_frame = ttk.Frame(right)
    mode_select_frame.pack(fill=tk.X)
    ttk.Label(mode_select_frame, text="FFT Mode").pack(side=tk.LEFT)
    fft_mode_var = tk.IntVar(value=1)
    fft_mode_combo = ttk.Combobox(mode_select_frame, values=[], state="readonly", width=12)
    fft_mode_combo.pack(side=tk.LEFT, padx=4)

    ttk.Label(mode_select_frame, text="Mode").pack(side=tk.LEFT)
    cloud_mode_var = tk.IntVar(value=1)
    cloud_mode_combo = ttk.Combobox(mode_select_frame, values=[], state="readonly", width=12)
    cloud_mode_combo.pack(side=tk.LEFT, padx=4)

    ttk.Label(mode_select_frame, text="Variable").pack(side=tk.LEFT)
    cloud_var_combo = ttk.Combobox(mode_select_frame, values=[], state="readonly", width=12)
    cloud_var_combo.pack(side=tk.LEFT, padx=4)

    ttk.Label(mode_select_frame, text="Z slice").pack(side=tk.LEFT)
    z_slice_var = tk.StringVar(value="Auto")
    z_slice_combo = ttk.Combobox(mode_select_frame, values=["Auto"], textvariable=z_slice_var, state="readonly", width=10)
    z_slice_combo.pack(side=tk.LEFT, padx=4)

    def _on_mode_select(event=None):  # noqa: ANN001
        val = cloud_mode_combo.get().strip()
        if val.lower().startswith("mode"):
            try:
                idx = int(val.split()[1])
                cloud_mode_var.set(idx)
                update_plots()
            except Exception:
                return

    def _on_var_select(event=None):  # noqa: ANN001
        update_plots()

    def _on_fft_select(event=None):  # noqa: ANN001
        val = fft_mode_combo.get().strip()
        if val.lower().startswith("mode"):
            try:
                idx = int(val.split()[1])
                fft_mode_var.set(idx)
            except Exception:
                pass
        update_plots()

    cloud_mode_combo.bind("<<ComboboxSelected>>", _on_mode_select)
    fft_mode_combo.bind("<<ComboboxSelected>>", _on_fft_select)
    cloud_var_combo.bind("<<ComboboxSelected>>", _on_var_select)
    z_slice_combo.bind("<<ComboboxSelected>>", _on_var_select)

    # Log (shorter height)
    log_frame = ttk.LabelFrame(right, text="Log")
    log_frame.pack(fill=tk.X)
    log_text = tk.Text(log_frame, height=6, state="disabled")
    log_text.pack(fill=tk.X)

    root.mainloop()


if __name__ == "__main__":
    run_tk_ui()
