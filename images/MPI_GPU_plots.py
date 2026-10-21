import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
import os

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.8,
    "lines.markersize": 6,
})

# Couleurs
C_BLO = "#1f77b4"      # Bleu (MPI Blocking)
C_NONBLO = "#d62728"   # Rouge (MPI Non-Blocking)
C_GPU = "#2ca02c"      # Vert (GPU)
C_CPU = "#ff7f0e"      # Orange (CPU Seq)
C_IDEAL = "#7f7f7f"    # Gris (Lignes idéales)
C_BEST = "#e377c2"     # Rose (Meilleur bloc)
GRID_KW = dict(linestyle=":", alpha=0.5)

# Dossier de sortie
OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)

def save_fig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.tight_layout()
    fig.savefig(f"{path}.pdf", bbox_inches="tight")
    fig.savefig(f"{path}.png", bbox_inches="tight")
    print(f"Saved: {path}.[pdf|png]")

def plot_mpi_figures(csv_file):
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: {csv_file} not found. Skipping MPI plots.")
        return

    images = df['image'].unique()
    
    for img in images:
        df_img = df[df['image'] == img]
        
        # Determine fixed parameters dynamically
        p_values = sorted(df_img['p'].unique())
        nbiter_values = sorted(df_img['nbiter'].unique())
        
        # Typically, we fix nbiter to the max for the 'vs p' plots
        # and we fix p to the max for the 'vs nbiter' plots
        if img == 'femme10':
            iter_fixed = 1000000
            p_fixed = 32
            min_p = 1
        else:
            iter_fixed = 2000
            p_fixed = 32
            min_p = min(p_values) # Usually 8 for Sukhothai

        print(f"\n--- Generating MPI plots for {img} ---")

        df_p = df_img[df_img['nbiter'] == iter_fixed]
        
        if not df_p.empty:
            df_blo = df_p[df_p['algo'] == 'blo'].sort_values('p')
            df_nonblo = df_p[df_p['algo'] == 'nonblo'].sort_values('p')
            
            p_blo = df_blo['p'].values
            t_blo = df_blo['time_s'].values
            p_nonblo = df_nonblo['p'].values
            t_nonblo = df_nonblo['time_s'].values

            if len(t_blo) > 0 and len(t_nonblo) > 0:
                t_base_blo = t_blo[0]
                t_base_nonblo = t_nonblo[0]
                
                sp_blo = t_base_blo / t_blo
                sp_nonblo = t_base_nonblo / t_nonblo
                
                # Efficiency relative to min_p
                eff_blo = sp_blo / (p_blo / min_p)
                eff_nonblo = sp_nonblo / (p_nonblo / min_p)

                fig, axes = plt.subplots(2, 1, figsize=(8, 10))
                title_suffix = f"({img}, nbiter={iter_fixed})" if min_p == 1 else f"({img}, relative to p={min_p})"
                fig.suptitle(f"MPI Performance vs p {title_suffix}", fontweight="bold")
                
                # Speedup
                if min_p == 1:
                    axes[0].plot(p_blo, p_blo, "--", color=C_IDEAL, label="Ideal")
                else:
                    axes[0].plot(p_blo, p_blo / min_p, "--", color=C_IDEAL, label=f"Ideal (rel. p={min_p})")
                
                axes[0].plot(p_blo, sp_blo, "o-", color=C_BLO, label="Blocking")
                axes[0].plot(p_nonblo, sp_nonblo, "s--", color=C_NONBLO, label="Non-blocking")
                axes[0].set(xlabel="Processes $p$", ylabel="Speedup $S(p)$", xscale="log", yscale="log")
                axes[0].set_xticks(p_values); axes[0].xaxis.set_major_formatter(ticker.ScalarFormatter()); axes[0].yaxis.set_major_formatter(ticker.ScalarFormatter())
                axes[0].legend(); axes[0].grid(**GRID_KW)

                # Efficiency
                axes[1].axhline(1.0, linestyle="--", color=C_IDEAL, label="Ideal")
                axes[1].plot(p_blo, eff_blo, "o-", color=C_BLO, label="Blocking")
                axes[1].plot(p_nonblo, eff_nonblo, "s--", color=C_NONBLO, label="Non-blocking")
                axes[1].set(xlabel="Processes $p$", ylabel="Efficiency $E(p)$", xscale="log", ylim=(0, 1.1))
                axes[1].set_xticks(p_values); axes[1].xaxis.set_major_formatter(ticker.ScalarFormatter())
                axes[1].legend(); axes[1].grid(**GRID_KW)
                
                save_fig(fig, f"fig_mpi_speedup_vs_p_{img}")

                # Match p values to compute gain
                gain_p_vals = []
                gain_pct = []
                for p_val in p_values:
                    t_b = df_blo[df_blo['p'] == p_val]['time_s'].values
                    t_nb = df_nonblo[df_nonblo['p'] == p_val]['time_s'].values
                    if len(t_b) > 0 and len(t_nb) > 0:
                        gain_p_vals.append(p_val)
                        gain_pct.append(((t_b[0] - t_nb[0]) / t_b[0]) * 100)

                if gain_p_vals:
                    fig, ax = plt.subplots(figsize=(7, 4.5))
                    bars = ax.bar([str(p) for p in gain_p_vals], gain_pct, color=C_NONBLO, alpha=0.8, edgecolor="white")
                    ax.axhline(0, color="black", linewidth=1)
                    ax.set(xlabel="Processes $p$", ylabel="Gain (%)", title=f"Gain of Non-blocking over Blocking ({img}, nbiter={iter_fixed})")
                    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
                    ax.grid(axis='y', **GRID_KW)
                    save_fig(fig, f"fig_mpi_gain_vs_p_{img}")

        df_iter = df_img[df_img['p'] == p_fixed]
        if not df_iter.empty:
            df_blo_i = df_iter[df_iter['algo'] == 'blo'].sort_values('nbiter')
            df_nb_i = df_iter[df_iter['algo'] == 'nonblo'].sort_values('nbiter')
            
            fig, ax = plt.subplots(figsize=(7, 4.5))
            
            # Noise region (below 1000 iterations generally)
            ax.axvspan(min(nbiter_values), 1000, color="gray", alpha=0.15, label="Fixed-cost regime (nbiter < 1000)")
            
            ax.plot(df_blo_i['nbiter'], df_blo_i['time_s'], "o-", color=C_BLO, label="Blocking")
            ax.plot(df_nb_i['nbiter'], df_nb_i['time_s'], "s--", color=C_NONBLO, label="Non-blocking")
            
            ax.set(xlabel="Number of iterations", ylabel="Time (s)", xscale="log", yscale="log", 
                   title=f"Execution Time vs Iterations ({img}, p={p_fixed})")
            ax.legend(); ax.grid(**GRID_KW)
            save_fig(fig, f"fig_mpi_time_vs_iter_{img}")


def plot_gpu_figures(csv_file):
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: {csv_file} not found. Skipping GPU plots.")
        return

    # Calculate speedup directly
    df['speedup'] = df['cpu_time_s'] / df['gpu_time_s']
    images = df['image'].unique()

    for img in images:
        df_img = df[df['image'] == img]
        print(f"\n--- Generating GPU plots for {img} ---")

        if img == 'femme10':
            fixed_iter_block = 10000
        else:
            fixed_iter_block = 200

        df_block = df_img[(df_img['nbiter'] == fixed_iter_block) & (df_img['check'] == 'ok')]
        if not df_block.empty:
            all_bx = sorted(df_block['bx'].unique())
            all_by = sorted(df_block['by'].unique())
            
            # Create a matrix for the heatmap
            mat = np.full((len(all_by), len(all_bx)), np.nan)
            for _, row in df_block.iterrows():
                i = all_by.index(row['by'])
                j = all_bx.index(row['bx'])
                mat[i, j] = row['gpu_time_s'] * 1000 # to ms

            fig, ax = plt.subplots(figsize=(9, 4.5))
            im = ax.imshow(mat, aspect="auto", cmap="RdYlGn_r", norm=mcolors.LogNorm(vmin=np.nanmin(mat), vmax=np.nanmax(mat)))
            ax.set_xticks(range(len(all_bx))); ax.set_xticklabels([str(b) for b in all_bx])
            ax.set_yticks(range(len(all_by))); ax.set_yticklabels([str(b) for b in all_by])
            ax.set(xlabel="bx (threads in x)", ylabel="by (threads in y)", title=f"GPU Time (ms) by Block Shape ({img}, nbiter={fixed_iter_block})")
            
            # Annotate cells
            for i in range(len(all_by)):
                for j in range(len(all_bx)):
                    if not np.isnan(mat[i, j]): 
                        ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=7, color="black")
            
            # Highlight best
            best_i, best_j = np.unravel_index(np.nanargmin(mat), mat.shape)
            ax.add_patch(plt.Rectangle((best_j - 0.5, best_i - 0.5), 1, 1, fill=False, edgecolor=C_BEST, linewidth=2.5, 
                                       label=f"Best: {all_bx[best_j]}x{all_by[best_i]}"))
            
            plt.colorbar(im, ax=ax, label="Time (ms)")
            ax.legend(loc="upper right", fontsize=8)
            save_fig(fig, f"fig_gpu_heatmap_{img}")

            df_block.loc[:, 'total_threads'] = df_block['bx'] * df_block['by']
            # Group by total_threads to find the max speedup for each size
            df_max_sp = df_block.groupby('total_threads')['speedup'].max().reset_index()
            df_max_sp = df_max_sp.sort_values('total_threads')

            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.plot(df_max_sp['total_threads'], df_max_sp['speedup'], "o-", color=C_GPU, label="Max GPU Speedup")
            ax.axhline(1.0, linestyle="--", color=C_CPU, label="CPU Reference (1.0x)")
            
            ax.set(xlabel="Total threads per block (bx × by)", ylabel="Speedup (GPU / CPU)", 
                   title=f"Best Speedup vs Total Threads ({img}, nbiter={fixed_iter_block})", xscale="log")
            
            # **CORRECTION**: Use set_xscale with basex or base, but matplotlib updated to base=2.
            # To be safe across versions, setting base directly or formatting:
            ax.set_xscale("log", base=2)
            ax.set_xticks(df_max_sp['total_threads'])
            ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
            ax.legend(); ax.grid(**GRID_KW)
            save_fig(fig, f"fig_gpu_speedup_vs_threads_{img}")

        df_iter = df_img[(df_img['bx'] == 16) & (df_img['by'] == 16)].sort_values('nbiter')
        if not df_iter.empty and len(df_iter) > 1:
            iters = df_iter['nbiter'].values
            t_gpu = df_iter['gpu_time_s'].values
            t_cpu = df_iter['cpu_time_s'].values
            speedup_iter = df_iter['speedup'].values
            
            # Subplot 1: Speedup vs nbiter
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.plot(iters, speedup_iter, "o-", color=C_GPU)
            ax.axhline(1.0, linestyle="--", color=C_CPU, label="No speedup")
            ax.set(xlabel="Number of iterations", ylabel="Speedup (GPU / CPU)", xscale="log", 
                   title=f"GPU Speedup vs Iterations ({img}, block 16x16)")
            ax.legend(); ax.grid(**GRID_KW)
            save_fig(fig, f"fig_gpu_speedup_vs_iter_{img}")

            # Subplot 2: Time per iteration (Amortization)
            t_per_iter_gpu = (t_gpu / iters) * 1e6 # in micro-seconds
            t_per_iter_cpu = (t_cpu / iters) * 1e6
            
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.plot(iters, t_per_iter_gpu, "o-", color=C_GPU, label="GPU (16x16)")
            ax.plot(iters, t_per_iter_cpu, "s--", color=C_CPU, label="CPU Reference")
            ax.axvspan(1, 100, color="gray", alpha=0.15, label="Launch overhead dominated")
            
            # **CORRECTION**: Use raw strings (r"...") for LaTeX syntax \mu
            ax.set(xlabel="Number of iterations", ylabel=r"Time per iteration ($\mu$s)", xscale="log", yscale="log",
                   title=f"Overhead Amortization ({img})")
            ax.legend(); ax.grid(**GRID_KW)
            save_fig(fig, f"fig_gpu_amortization_{img}")

def plot_mpi_vs_gpu(mpi_csv, gpu_csv):
    try:
        df_mpi = pd.read_csv(mpi_csv)
        df_gpu = pd.read_csv(gpu_csv)
    except FileNotFoundError:
        print("Missing CSV files for comparison. Skipping.")
        return

    # The GPU nbiter sweep of the large image was run on the 4080x6132 version
    # (25.0 Mpx, 0.8% fewer pixels than 4096x6156, which MPI needs because h must
    # be divisible by p=32). We pair the MPI image with its closest GPU image.
    GPU_IMAGE_FOR = {"Sukhothai_4096x6156": "Sukhothai_4080x6132"}

    for img in df_mpi['image'].unique():
        gpu_img = img if img in set(df_gpu['image'].unique()) else GPU_IMAGE_FOR.get(img)
        if gpu_img is None:
            continue
        # prefer the GPU image with the richest nbiter sweep
        if img in GPU_IMAGE_FOR and (df_gpu['image'] == img).sum() > 0:
            n_own = df_gpu[(df_gpu['image'] == img) & (df_gpu['bx'] == 16) & (df_gpu['by'] == 16)]['nbiter'].nunique()
            n_alt = df_gpu[(df_gpu['image'] == GPU_IMAGE_FOR[img]) & (df_gpu['bx'] == 16) & (df_gpu['by'] == 16)]['nbiter'].nunique()
            gpu_img = img if n_own >= n_alt else GPU_IMAGE_FOR[img]
        print(f"\n--- Generating MPI vs GPU Comparison for {img} (GPU image: {gpu_img}) ---")

        # MPI Non-blocking at p=32
        df_mpi_img = df_mpi[(df_mpi['image'] == img) & (df_mpi['p'] == 32) & (df_mpi['algo'] == 'nonblo')].sort_values('nbiter')
        # GPU at block 16x16
        df_gpu_img = df_gpu[(df_gpu['image'] == gpu_img) & (df_gpu['bx'] == 16) & (df_gpu['by'] == 16)].sort_values('nbiter')

        if not df_mpi_img.empty and not df_gpu_img.empty:
            iters_mpi = df_mpi_img['nbiter'].values
            t_per_iter_mpi = (df_mpi_img['time_s'].values / iters_mpi) * 1e6
            iters_gpu = df_gpu_img['nbiter'].values
            t_per_iter_gpu = (df_gpu_img['gpu_time_s'].values / iters_gpu) * 1e6

            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.axvspan(1, 1000, color="gray", alpha=0.15, label="Fixed-cost regime")
            ax.plot(iters_mpi, t_per_iter_mpi, "s--", color=C_NONBLO, label="MPI Non-Blocking (32 cores)")
            ax.plot(iters_gpu, t_per_iter_gpu, "o-", color=C_GPU, label="GPU Tesla P100 (16x16)")
            ttl = f"MPI (32 cores) vs GPU: Time per Iteration ({img})"
            if gpu_img != img:
                ttl += f"\nGPU measured on {gpu_img}"
            ax.set(xlabel="Number of iterations", ylabel=r"Time per iteration ($\mu$s)", xscale="log", yscale="log", title=ttl)
            ax.legend(); ax.grid(**GRID_KW)
            save_fig(fig, f"fig_compare_mpi_vs_gpu_{img}")


if __name__ == "__main__":
    print("="*50)
    print(" GENERATING ALL HPC PLOTS")
    print("="*50)
    
    # Files
    f_cpu = "results_CPU.csv"
    f_gpu = "results_gpu.csv"

    plot_mpi_figures(f_cpu)
    plot_gpu_figures(f_gpu)
    plot_mpi_vs_gpu(f_cpu, f_gpu)
    
    print("\nAll done! Check the 'figures/' directory.")