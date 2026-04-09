import cuda.tile as ct
from cuda.tile import RoundingMode as RMd   
import torch
import torch.nn.functional as F
import math
from utils import benchmark, plot_benchmarks, time_fn

BLOCK_M = 64  # tile size for Q rows (query dimension)
BLOCK_N = 64  # tile size for K/V rows (key dimension)
INV_LOG_2 = 1.0 / math.log(2)


def pytorch_manual_attention(Q, K, V):
    # attention(q,k,v) = softmax(Q @ K^T / sqrt(d_k)) @ V
    # d_k -> dimension of the key vectors (HEAD_DIM in this code)
    # softmax x_i = e^x_i / sum (j = 1->N) (e^x_j)
    return F.softmax(( Q @  K.transpose(-2, -1) )/ math.sqrt(K.shape[-1]), dim=-1) @ V

def pytorch_attention(Q, K, V):
    return F.scaled_dot_product_attention(Q, K, V)

@ct.kernel
def cutile_attention_v1_kernel(Q: ct.Array,
                               K: ct.Array,
                               V: ct.Array,
                               scale: float,
                               SEQ: ct.Constant[int],
                               HEAD_DIM: ct.Constant[int],
                               O: ct.Array # output,
                               ):
    """
    Naive self-attention kernel.
    Tiles Q rows only, loads full K and V per tile.
    Does not scale to long sequences (full K and V must fit in SRAM).   
    """
    
    batch_idx = ct.bid(0)
    head_idx = ct.bid(1)
    query_row_idx = ct.bid(2)

    q = ct.load(Q, # loading from tensor Q
                (batch_idx, head_idx, query_row_idx, 0), # tile index
                (1, 1, BLOCK_M, HEAD_DIM) # how much to load (1 batch, 1 head, BLOCK_M rows/SEQ, full HEAD_DIM)
                )

    k = ct.load(K, # loading from tensor K
            (batch_idx, head_idx, 0, 0), # tile index
            (1, 1, SEQ, HEAD_DIM), # how much to load (1 batch, 1 head, full SEQ, full HEAD_DIM)
            )
    
    # squeeze tensors (to remove batch and head dims)
    q = ct.reshape(q, (BLOCK_M, HEAD_DIM))  

    k = ct.reshape(k, (SEQ, HEAD_DIM))
    k_t = ct.transpose(k)

    
    # matmul
    acc = ct.zeros((BLOCK_M, SEQ), dtype=ct.float32)
    acc = ct.mma(q,k_t, acc) # (BLOCK_M, HEAD_DIM) @ (HEAD_DIM, SEQ) -> (BLOCK_M, SEQ)
    
    # scale
    acc = acc * scale

    # softmax, for numerical stability we subtract the row max
    # softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))      
    m = ct.max(acc, axis=-1, keepdims=True)
    acc = acc - m
    acc = ct.exp(acc)
    denom = ct.sum(acc, axis=-1, keepdims=True)
    acc = acc / denom


    # final matmul
    v = ct.load(V, # loading from tensor V
            (batch_idx, head_idx, 0, 0), # tile index
            (1, 1, SEQ, HEAD_DIM) # how much to load (1 batch, 1 head, full SEQ, full HEAD_DIM)
            ) 
    v = ct.reshape(v, ((SEQ, HEAD_DIM)))
    v = ct.astype(v, ct.float32)
    
    out = ct.zeros((BLOCK_M, HEAD_DIM), dtype=ct.float32)
    out = ct.mma(acc, v, out) # (BLOCK_M, SEQ) @ (SEQ, HEAD_DIM) -> (BLOCK_M, HEAD_DIM)

    ct.store(O, # output mem
             (batch_idx, head_idx, query_row_idx, 0), # tile index
             ct.reshape(ct.astype(out, ct.float16), (1, 1, BLOCK_M, HEAD_DIM)) # cutile array (unsqueezed) and cast back to float16
             )
def cutile_attention_v1(Q, K, V):
    batch, num_heads, seq, head_dim = Q.shape
    stream = torch.cuda.current_stream().cuda_stream
    grid = (batch, num_heads, math.ceil(seq/BLOCK_M))
    O = torch.empty_like(Q)
    scale = 1.0 / math.sqrt(head_dim)
    ct.launch(stream, grid, cutile_attention_v1_kernel, (Q, K, V, scale, seq, head_dim, O))
    return O


@ct.kernel
def cutile_attention_v2_kernel(Q: ct.Array,
                               K: ct.Array,
                               V: ct.Array,
                               S: ct.Array, # score matrix (BATCH, NUM_HEADS, SEQ, SEQ) to store all values of Q @ K^T
                               scale: float,
                               SEQ: ct.Constant[int],
                               HEAD_DIM: ct.Constant[int],
                               O: ct.Array # output,
                               ):
    """
    Naive self-attention kernel but a bit smarter. 
    Tiles Q, K, V
    A lot of back and forth between HBM and SRAM but we are now able to support bigger K and V sizes (very very slow)
    """
    
    batch_idx = ct.bid(0)
    head_idx = ct.bid(1)
    query_row_idx = ct.bid(2)

    q = ct.load(Q, # loading from tensor Q
                (batch_idx, head_idx, query_row_idx, 0), # tile index
                (1, 1, BLOCK_M, HEAD_DIM) # how much to load (1 batch, 1 head, BLOCK_M rows/SEQ, full HEAD_DIM)
                )
    q = ct.reshape(q, (BLOCK_M, HEAD_DIM))  

    for j in range((SEQ + BLOCK_N - 1) // BLOCK_N ):
        k = ct.load(K, # loading from tensor K
            (batch_idx, head_idx, j, 0), # tile index
            (1, 1, BLOCK_N, HEAD_DIM), # how much to load (1 batch, 1 head, BLOCK_N rows/SEQ, full HEAD_DIM)
            ) 
        # squeeze tensors (to remove batch and head dims)
        k = ct.reshape(k, (BLOCK_N, HEAD_DIM))
        k_t = ct.transpose(k)  


    
        # matmul
        acc = ct.zeros((BLOCK_M, BLOCK_N), dtype=ct.float32)
        acc = ct.mma(q,k_t, acc) # (BLOCK_M, HEAD_DIM) @ (HEAD_DIM, BLOCK_N) -> (BLOCK_M, BLOCK_N)
        # scale
        acc = acc * scale

        # store results in HBM
        ct.store(S, # score mem
                (batch_idx, head_idx, query_row_idx, j), # tile index
                ct.reshape(ct.astype(acc, ct.float32), (1, 1, BLOCK_M, BLOCK_N)) # cutile array (unsqueezed) and keep in float32 since it's an intermediate value (even if acc is already float32, kept it for clarity)
                )


    # softmax, for numerical stability we subtract the row max
    # softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))    

    # find row max   
    row_max = ct.full((BLOCK_M, 1), float("-inf"), dtype=ct.float32)                                 
    for j in range((SEQ + BLOCK_N - 1) // BLOCK_N ):                                                                                                                                   
        s_tile = ct.load(S, # loading from tensor S
                        (batch_idx, head_idx, query_row_idx, j), # tile index
                        (1, 1, BLOCK_M, BLOCK_N) # how much to load (1 batch, 1 head, BLOCK_M rows/SEQ, BLOCK_N rows/SEQ)
                        )    
        s_tile = ct.reshape(s_tile, (BLOCK_M, BLOCK_N))    
        row_max = ct.maximum(row_max, ct.max(s_tile, axis=-1, keepdims=True)) # max per row (reduce), then max element wise with (BLOCK_M, 1) that is getting updated after each block
                             

    # exp and row sum
    row_sum = ct.zeros((BLOCK_M, 1), dtype=ct.float32)   
    for j in range((SEQ + BLOCK_N - 1) // BLOCK_N ):      
        s_tile = ct.load(S, # loading from tensor S
                (batch_idx, head_idx, query_row_idx, j), # tile index
                (1, 1, BLOCK_M, BLOCK_N) # how much to load (1 batch, 1 head, BLOCK_M rows/SEQ, BLOCK_N rows/SEQ)
                )
        s_tile = ct.reshape(s_tile, (BLOCK_M, BLOCK_N)) 
        s_tile = ct.exp(s_tile - row_max)
        row_sum = row_sum + ct.sum(s_tile, axis=-1, keepdims=True)

        # store results in HBM
        ct.store(S, # score mem
                (batch_idx, head_idx, query_row_idx, j), # tile index
                ct.reshape(s_tile, (1, 1, BLOCK_M, BLOCK_N)) # cutile array (unsqueezed) and keep in float32 since it's an intermediate value (even if acc is already float32, kept it for clarity)
                )
        
    # noramlize
    for j in range((SEQ + BLOCK_N - 1) // BLOCK_N ):      
        s_tile = ct.load(S, # loading from tensor S
                (batch_idx, head_idx, query_row_idx, j), # tile index
                (1, 1, BLOCK_M, BLOCK_N) # how much to load (1 batch, 1 head, BLOCK_M rows/SEQ, BLOCK_N rows/SEQ)
                )
        s_tile = ct.reshape(s_tile, (BLOCK_M, BLOCK_N))
        s_tile = s_tile / row_sum

        # store results in HBM
        ct.store(S, # score mem
                (batch_idx, head_idx, query_row_idx, j), # tile index
                ct.reshape(s_tile, (1, 1, BLOCK_M, BLOCK_N)) # cutile array (unsqueezed) and keep in float32 since it's an intermediate value (even if acc is already float32, kept it for clarity)
                )

    # final matmul
    acc = ct.zeros((BLOCK_M, HEAD_DIM), dtype=ct.float32)
    for j in range((SEQ + BLOCK_N - 1) // BLOCK_N ):    
        v = ct.load(V, # loading from tensor V
                (batch_idx, head_idx, j, 0), # tile index
                (1, 1, BLOCK_N, HEAD_DIM), # how much to load (1 batch, 1 head, BLOCK_N rows/SEQ, full HEAD_DIM)
                )
        v = ct.reshape(v, ((BLOCK_N, HEAD_DIM)))
        v = ct.astype(v, ct.float32)

        s_tile = ct.load(S, # loading from tensor S
                (batch_idx, head_idx, query_row_idx, j), # tile index
                (1, 1, BLOCK_M, BLOCK_N) # how much to load (1 batch, 1 head, BLOCK_M rows/SEQ, BLOCK_N rows/SEQ)
                )
        s_tile = ct.reshape(s_tile, (BLOCK_M, BLOCK_N))
    
        acc = ct.mma(s_tile, v, acc) # (BLOCK_M, BLOCK_N) @ (BLOCK_N, HEAD_DIM) -> (BLOCK_M, HEAD_DIM)

    ct.store(O, # output mem
             (batch_idx, head_idx, query_row_idx, 0), # tile index
             ct.reshape(ct.astype(acc, ct.float16), (1, 1, BLOCK_M, HEAD_DIM)) # cutile array (unsqueezed) and cast back to float16
             )
def cutile_attention_v2(Q, K, V):
    batch, num_heads, seq, head_dim = Q.shape
    stream = torch.cuda.current_stream().cuda_stream
    grid = (batch, num_heads, math.ceil(seq/BLOCK_M))
    O = torch.empty_like(Q)
    S = torch.zeros(batch, num_heads, seq, seq, device="cuda")
    scale = 1.0 / math.sqrt(head_dim)
    ct.launch(stream, grid, cutile_attention_v2_kernel, (Q, K, V, S, scale, seq, head_dim, O))
    return O

@ct.kernel
def cutile_flash_attention_v1_kernel(Q: ct.Array,
                                     K: ct.Array,
                                     V: ct.Array,
                                     scale: float,
                                     SEQ: ct.Constant[int],
                                     HEAD_DIM: ct.Constant[int],
                                     O: ct.Array # output,
                                    ):
    """
    Time for flash attention making full use of the online softmax trick.
    1. Compute scores for new K tile
    2. Find this tile's max
    3. Compare with running max --> did the max change?
    4. correct old o and l
    5. Compute exp(scores - new_max)
    6. Accumulate into l and o 
    """
    
    batch_idx = ct.bid(0)
    head_idx = ct.bid(1)
    query_row_idx = ct.bid(2)

    q = ct.load(Q, # loading from tensor Q
                (batch_idx, head_idx, query_row_idx, 0), # tile index
                (1, 1, BLOCK_M, HEAD_DIM) # how much to load (1 batch, 1 head, BLOCK_M rows/SEQ, full HEAD_DIM)
                )
    q = ct.reshape(q, (BLOCK_M, HEAD_DIM))


    # running max
    m = ct.full((BLOCK_M, 1), float("-inf"), dtype=ct.float32)
    # running sum
    l = ct.zeros((BLOCK_M, 1), dtype=ct.float32)
    # output accumulator
    o = ct.zeros((BLOCK_M, HEAD_DIM), dtype=ct.float32)

    for j in range((SEQ + BLOCK_N - 1) // BLOCK_N ):
        k = ct.load(K, # loading from tensor K
            (batch_idx, head_idx, j, 0), # tile index
            (1, 1, BLOCK_N, HEAD_DIM), # how much to load (1 batch, 1 head, BLOCK_N rows/SEQ, full HEAD_DIM)
            )

        v = ct.load(V, # loading from tensor V
            (batch_idx, head_idx, j, 0), # tile index
            (1, 1, BLOCK_N, HEAD_DIM), # how much to load (1 batch, 1 head, BLOCK_N rows/SEQ, full HEAD_DIM)
            )
        v = ct.astype(v, ct.float32)

        # squeeze tensors (to remove batch and head dims)
        k = ct.reshape(k, (BLOCK_N, HEAD_DIM))
        k_t = ct.transpose(k)
        # squeeze tensors (to remove batch and head dims)
        v = ct.reshape(v, ((BLOCK_N, HEAD_DIM)))


        # matmul
        acc = ct.zeros((BLOCK_M, BLOCK_N), dtype=ct.float32)
        acc = ct.mma(q,k_t, acc) # (BLOCK_M, HEAD_DIM) @ (HEAD_DIM, BLOCK_N) -> (BLOCK_M, BLOCK_N)
        # scale
        acc = acc * scale

        # online softmax: maximum of tile
        m_tile = ct.max(acc, axis=-1, keepdims=True)
        m_new = ct.maximum(m, m_tile)

        # correction factor: rescale previous results to use new max
        correction = ct.exp(m - m_new)
        l = correction * l # l originally contains sum(exp(s - m_old)), correction rescales to sum(exp(s - m_new))
        o = correction * o # o originally contains o = exp(s0 - m_old) @ V0, so after correction -> exp(s0 - m_new) @ V0 = exp(s0 - m_old) * exp(m_old - m_new) @ V0:

        # accumulate this tile's contribution
        softmax_numerator_tile = ct.exp(acc - m_new)
        l = l + ct.sum(softmax_numerator_tile, axis=-1, keepdims=True)

        o = ct.mma(softmax_numerator_tile, v, o)

        m = m_new

    # normalize by full row sum
    o = o / l

    ct.store(O, # output mem
             (batch_idx, head_idx, query_row_idx, 0), # tile index
             ct.reshape(ct.astype(o, ct.float16), (1, 1, BLOCK_M, HEAD_DIM)) # cutile array (unsqueezed) and cast back to float16
             )
def cutile_flash_attention_v1(Q, K, V):
    batch, num_heads, seq, head_dim = Q.shape
    stream = torch.cuda.current_stream().cuda_stream
    grid = (batch, num_heads, math.ceil(seq/BLOCK_M))
    O = torch.empty_like(Q)
    scale = 1.0 / math.sqrt(head_dim)
    ct.launch(stream, grid, cutile_flash_attention_v1_kernel, (Q, K, V, scale, seq, head_dim, O))
    return O

@ct.kernel(occupancy=2)
def cutile_flash_attention_v2_kernel(Q: ct.Array,
                                     K: ct.Array,
                                     V: ct.Array,
                                     qk_scale: float,
                                     SEQ: ct.Constant[int],
                                     HEAD_DIM: ct.Constant[int],
                                     BLOCK_M: ct.Constant[int],
                                     BLOCK_N: ct.Constant[int],
                                     O: ct.Array # output,
                                    ):
    """
    Just an optimized version of the v1 flash attention kernel
    """
    
    batch_idx = ct.bid(0)
    head_idx = ct.bid(1)
    query_row_idx = ct.bid(2)

    q = ct.load(Q, # loading from tensor Q
                (batch_idx, head_idx, query_row_idx, 0), # tile index
                (1, 1, BLOCK_M, HEAD_DIM) # how much to load (1 batch, 1 head, BLOCK_M rows/SEQ, full HEAD_DIM)
                )
    q = ct.reshape(q, (BLOCK_M, HEAD_DIM))  


    # running max
    m = ct.full((BLOCK_M, 1), float("-inf"), dtype=ct.float32)
    # running sum
    l = ct.zeros((BLOCK_M, 1), dtype=ct.float32)
    # output accumulator
    o = ct.zeros((BLOCK_M, HEAD_DIM), dtype=ct.float32)

    # pre-scale for exp2 inside kernel
    qk_scale = qk_scale * INV_LOG_2

    for j in range(ct.cdiv(SEQ, BLOCK_N)):
        # load K transposed at load time via order=(0,1,3,2), no ct.transpose needed
        k_t = ct.load(K,
            (batch_idx, head_idx, 0, j), # note: dim 2 is 0, dim 3 is j (transposed indexing)
            (1, 1, HEAD_DIM, BLOCK_N), # transposed shape
            order=(0, 1, 3, 2),
            latency=2)
        k_t = ct.reshape(k_t, (HEAD_DIM, BLOCK_N))

        v = ct.load(V, # loading from tensor V
            (batch_idx, head_idx, j, 0), # tile index
            (1, 1, BLOCK_N, HEAD_DIM), # how much to load (1 batch, 1 head, BLOCK_N rows/SEQ, full HEAD_DIM)
            latency=4)
        v = ct.reshape(v, ((BLOCK_N, HEAD_DIM)))


        # matmul
        acc = ct.mma(q, k_t, ct.full((BLOCK_M, BLOCK_N), 0.0, dtype=ct.float32))

        # online softmax: maximum of tile
        m_new = max(m, ct.max(acc, axis=-1, keepdims=True) * qk_scale)
        acc = acc * qk_scale - m_new

        # accumulate this tile's contribution
        softmax_numerator_tile = ct.exp2(acc, flush_to_zero=True)
        l_new = ct.sum(softmax_numerator_tile, axis=-1, keepdims=True)
        correction = ct.exp2(m - m_new, flush_to_zero=True)

        # correction factor: rescale previous results to use new max
        l = l * correction + l_new
        o = o * correction

        o = ct.mma(softmax_numerator_tile.astype(Q.dtype), v, o)

        m = m_new

    # normalize by full row sum
    o = ct.truediv(o, l, flush_to_zero=True, rounding_mode=RMd.APPROX)

    ct.store(O, # output mem
             (batch_idx, head_idx, query_row_idx, 0), # tile index
             ct.reshape(ct.astype(o, ct.float16), (1, 1, BLOCK_M, HEAD_DIM)) # cutile array (unsqueezed) and cast back to float16
             )
# tile configs per head_dim (power-of-two tile sizes that fit in SRAM)
_TILE_CONFIGS_BY_HEAD_DIM = {
    32:  ([64, 128], [32, 64]),
    64:  ([64, 128, 256], [32, 64, 128]),
    128: ([64, 128, 256], [32, 64, 128]),
    256: ([64, 128],      [32, 64]),
}
_autotune_cache = {}
def _launch_v2(Q, K, V, qk_scale, seq, head_dim, bm, bn):
    batch, num_heads = Q.shape[:2]
    stream = torch.cuda.current_stream().cuda_stream
    grid = (batch, num_heads, math.ceil(seq / bm))
    O = torch.empty_like(Q)
    ct.launch(stream, grid, cutile_flash_attention_v2_kernel, (Q, K, V, qk_scale, seq, head_dim, bm, bn, O))
    return O
def cutile_flash_attention_v2(Q, K, V):
    # extract tensor dimensions
    batch, num_heads, seq, head_dim = Q.shape

    # raw scale: 1/sqrt(d_k), kernel multiplies by 1/ln(2) internally for exp2
    qk_scale = 1.0 / math.sqrt(head_dim)

    # key to cache tile sizes based on (seq, head_dim) tuple, we only autotune if new config
    cache_key = (seq, head_dim)
    if cache_key not in _autotune_cache:

        # get tile sizes based on head_dim from _TILE_CONFIGS_BY_HEAD_DIM
        tile_ms, tile_ns = _TILE_CONFIGS_BY_HEAD_DIM.get(head_dim, ([64, 128], [32, 64]))
        configs = [(bm, bn) for bm in tile_ms for bn in tile_ns
                   if seq % bm == 0 and seq % bn == 0]
        if not configs:
            configs = [(64, 64)]  # fallback

        print(f"------[autotune] seq={seq} head_dim={head_dim}")
        best_time = float("inf")
        best_config = configs[0]
        # run time_fn for every tile size config and choose the best one
        for bm, bn in configs:
            t = time_fn(lambda q, k, v: _launch_v2(q, k, v, qk_scale, seq, head_dim, bm, bn), Q, K, V)
            print(f"-----------BLOCK_M={bm:3d} BLOCK_N={bn:3d} -> {t:.3f} ms")
            if t < best_time:
                best_time = t
                best_config = (bm, bn)
        print(f"------[autotune] best: BLOCK_M={best_config[0]} BLOCK_N={best_config[1]} ({best_time:.3f} ms)")
        _autotune_cache[cache_key] = best_config

    # get best tile sizes
    bm, bn = _autotune_cache[cache_key]
    return _launch_v2(Q, K, V, qk_scale, seq, head_dim, bm, bn)


# optional tri dao's flash attention implementations
try: #TODO: setup flash-attn-4 (did not work on sm120)
    from flash_attn import flash_attn_func
    def flash_attn_dao(Q, K, V):
        # flash_attn expects (batch, seq, heads, dim), we have (batch, heads, seq, dim)
        return flash_attn_func(Q.transpose(1,2), K.transpose(1,2), V.transpose(1,2)).transpose(1,2)
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False




def run_benchmarks():
    configs = [
        (4, 32, 128, 64),      # small
        (4, 32, 256, 64),      # small-medium
        (4, 32, 512, 64),      # medium
        (4, 32, 1024, 64),     # medium-large
        (4, 32, 2048, 64),     # large
        (4, 32, 4096, 64),     # very large
    ]

    if HAS_FLASH_ATTN:
        print("flash_attn (Tri Dao) found")
    else:
        print("flash_attn not installed, skipping")

    all_results = []
    config_labels = []

    for BATCH, NUM_HEADS, SEQ, HEAD_DIM in configs:
        print(f"\n{'='*60}")
        print(f"BATCH={BATCH}, NUM_HEADS={NUM_HEADS}, SEQ={SEQ}, HEAD_DIM={HEAD_DIM}")
        print(f"{'='*60}")

        Q = torch.randn(BATCH, NUM_HEADS, SEQ, HEAD_DIM, dtype=torch.float16, device="cuda")
        K = torch.randn(BATCH, NUM_HEADS, SEQ, HEAD_DIM, dtype=torch.float16, device="cuda")
        V = torch.randn(BATCH, NUM_HEADS, SEQ, HEAD_DIM, dtype=torch.float16, device="cuda")

        fns = {
            "PyTorch SDPA": pytorch_attention,
            "Flash v2 (ours)": cutile_flash_attention_v2,
            "Flash v1 (ours)": cutile_flash_attention_v1,
        }
        if HAS_FLASH_ATTN:
            fns["flash_attn (Dao)"] = flash_attn_dao
        # only include naive/manual for small SEQ
        if SEQ <= 128:
            fns["Cutile v1 (naive)"] = cutile_attention_v1
        if SEQ <= 2048:
            fns["Cutile v2 (naive)"] = cutile_attention_v2
            fns["PyTorch (manual)"] = pytorch_manual_attention

        results = benchmark(fns, Q, K, V, ref_fn=pytorch_attention)
        all_results.append(results)
        config_labels.append(f"B={BATCH}, H={NUM_HEADS}\nseq={SEQ}, d={HEAD_DIM}")

    plot_benchmarks(all_results, config_labels)

if __name__=="__main__":
    run_benchmarks()