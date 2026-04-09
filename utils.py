import torch
import matplotlib.pyplot as plt
import numpy as np

WARMUP = 10
ITERS = 50

def time_fn(fn, *args):
    for _ in range(WARMUP):
        fn(*args)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITERS):
        fn(*args)
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / ITERS  # ms per call


def compute_error(out, out_ref):
    err = (out - out_ref).abs().max().item()
    return err


def benchmark(fns, Q, K, V, ref_fn=None):
    ref_out = ref_fn(Q, K, V) if ref_fn else None

    results = []
    for name, fn in fns.items():
        err = "-"
        if ref_out is not None:
            try:
                out = fn(Q, K, V)
                err = f"{compute_error(out, ref_out):.6f}"
            except Exception as e:
                err = f"ERROR"
                results.append((name, err, "FAILED", None))
                continue

        try:
            t = time_fn(fn, Q, K, V)
            results.append((name, err, f"{t:.3f} ms", t))
        except Exception as e:
            results.append((name, err, "FAILED", None))

    # Print table
    name_w = max(len(r[0]) for r in results)
    err_w = max(len(r[1]) for r in results)
    print(f"\n{'Name':<{name_w}}  {'Error':<{err_w}}  {'Time'}")
    print("-" * (name_w + err_w + 15))
    for name, err, time_str, _ in results:
        print(f"{name:<{name_w}}  {err:<{err_w}}  {time_str}")

    return results


COLORS = {
    "PyTorch SDPA":        "#2196F3",
    "PyTorch (manual)":    "#90CAF9",
    "Flash v2 (ours)":     "#4CAF50",
    "Flash v1 (ours)":     "#FFC107",
    "flash_attn (Dao)":    "#9C27B0",
    "Cutile v1 (naive)":   "#FF7043",
    "Cutile v2 (naive)":   "#EF5350",
}

def plot_benchmarks(all_results, config_labels, save_path="docs/benchmark.png"):
    # collect all implementation names across configs
    all_names = []
    for results in all_results:
        for name, _, _, _ in results:
            if name not in all_names:
                all_names.append(name)

    n_configs = len(config_labels)
    fig, axes = plt.subplots(1, n_configs, figsize=(5 * n_configs, 6), sharey=False)
    if n_configs == 1:
        axes = [axes]

    fig.suptitle("Attention Kernel Benchmarks", fontsize=14, fontweight="bold", y=0.98)

    for ci, (ax, results, label) in enumerate(zip(axes, all_results, config_labels)):
        # filter out FAILED entries
        valid = [(r[0], r[3]) for r in results if r[3] is not None]
        if not valid:
            continue

        # sort by time (fastest first)
        sorted_pairs = sorted(valid, key=lambda x: x[1])
        names = [p[0] for p in sorted_pairs]
        times = [p[1] for p in sorted_pairs]

        colors = [COLORS.get(n, "#9E9E9E") for n in names]

        bars = ax.barh(range(len(names)), times, color=colors, edgecolor="white", linewidth=0.5, height=0.6)

        # label each bar with its time
        for bar, t in zip(bars, times):
            label_text = f"{t:.3f} ms"
            # put label inside or outside depending on bar width
            if t > max(times) * 0.3:
                ax.text(bar.get_width() * 0.5, bar.get_y() + bar.get_height() / 2,
                        label_text, ha='center', va='center', fontsize=8,
                        fontweight='bold', color='white')
            else:
                ax.text(bar.get_width() + max(times) * 0.02, bar.get_y() + bar.get_height() / 2,
                        label_text, ha='left', va='center', fontsize=8)

        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel("Time (ms)", fontsize=9)
        ax.set_title(label.replace('\n', ', '), fontsize=10, pad=10)
        ax.set_xscale("log")
        ax.grid(axis='x', alpha=0.2)
        ax.invert_yaxis()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"\nPlot saved to {save_path}")
    plt.close()

    # line chart: time vs seq length for each implementation
    plot_scaling(all_results, config_labels, save_path.replace(".png", "_scaling.png"))


def plot_scaling(all_results, config_labels, save_path="docs/benchmark_scaling.png"):
    """Line chart showing how each implementation scales with sequence length."""
    # extract seq lengths from config labels
    seq_lengths = []
    for label in config_labels:
        for part in label.replace('\n', ', ').split(','):
            part = part.strip()
            if part.startswith('seq='):
                seq_lengths.append(int(part.split('=')[1]))
                break

    # collect all implementation names
    all_names = []
    for results in all_results:
        for name, _, _, _ in results:
            if name not in all_names:
                all_names.append(name)

    fig, ax = plt.subplots(figsize=(10, 6))

    for name in all_names:
        seqs = []
        times = []
        for si, results in enumerate(all_results):
            for n, _, _, time_ms in results:
                if n == name and time_ms is not None:
                    seqs.append(seq_lengths[si])
                    times.append(time_ms)
                    break

        if len(seqs) < 2:
            continue

        color = COLORS.get(name, "#9E9E9E")
        ax.plot(seqs, times, 'o-', color=color, label=name, linewidth=2, markersize=5)

    ax.set_xlabel("Sequence Length", fontsize=11)
    ax.set_ylabel("Time (ms)", fontsize=11)
    ax.set_title("Attention Kernel Scaling by Sequence Length", fontsize=13, fontweight="bold")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=9, loc='upper left')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # nicer x ticks
    ax.set_xticks(sorted(set(seq_lengths)))
    ax.set_xticklabels([str(s) for s in sorted(set(seq_lengths))])

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Plot saved to {save_path}")
    plt.close()