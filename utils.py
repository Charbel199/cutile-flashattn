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
                err = f"ERROR: {e}"

        t = time_fn(fn, Q, K, V)
        results.append((name, err, f"{t:.3f} ms", t))

    # Print table
    name_w = max(len(r[0]) for r in results)
    err_w = max(len(r[1]) for r in results)
    print(f"\n{'Name':<{name_w}}  {'Error':<{err_w}}  {'Time'}")
    print("-" * (name_w + err_w + 15))
    for name, err, time_str, _ in results:
        print(f"{name:<{name_w}}  {err:<{err_w}}  {time_str}")

    return results


def plot_benchmarks(all_results, config_labels, save_path="docs/benchmark.png"):
    # collect all implementation names across configs
    all_names = []
    for results in all_results:
        for name, _, _, _ in results:
            if name not in all_names:
                all_names.append(name)

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(config_labels))
    width = 0.8 / len(all_names)

    for i, name in enumerate(all_names):
        times = []
        for results in all_results:
            t = None
            for n, _, _, time_ms in results:
                if n == name:
                    t = time_ms
                    break
            times.append(t if t is not None else 0)

        offset = (i - len(all_names) / 2 + 0.5) * width
        bars = ax.bar(x + offset, times, width, label=name)

        # label bars with ms values
        for bar, t in zip(bars, times):
            if t > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f'{t:.1f}', ha='center', va='bottom', fontsize=7)

    ax.set_xlabel('Configuration')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Attention Kernel Benchmarks')
    ax.set_xticks(x)
    ax.set_xticklabels(config_labels, fontsize=8)
    ax.legend(fontsize=8, loc='upper left')
    ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nPlot saved to {save_path}")
    plt.close()