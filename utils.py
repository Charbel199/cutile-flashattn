import torch

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