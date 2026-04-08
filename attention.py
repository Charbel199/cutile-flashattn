import cuda.tile as ct
from cuda.tile import RoundingMode as RMd   
import torch
import torch.nn.functional as F
import math
from utils import benchmark, plot_benchmarks

BLOCK_M = 64  # tile size for Q rows (query dimension)
BLOCK_N = 64  # tile size for K/V rows (key dimension)


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
    q = ct.astype(q, ct.float32)

    k = ct.load(K, # loading from tensor K
            (batch_idx, head_idx, 0, 0), # tile index
            (1, 1, SEQ, HEAD_DIM), # how much to load (1 batch, 1 head, full SEQ, full HEAD_DIM)
            )
    k = ct.astype(k, ct.float32)
    
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
    q = ct.astype(q, ct.float32)
    q = ct.reshape(q, (BLOCK_M, HEAD_DIM))  

    for j in range((SEQ + BLOCK_N - 1) // BLOCK_N ):
        k = ct.load(K, # loading from tensor K
            (batch_idx, head_idx, j, 0), # tile index
            (1, 1, BLOCK_N, HEAD_DIM), # how much to load (1 batch, 1 head, BLOCK_N rows/SEQ, full HEAD_DIM)
            ) 
        k = ct.astype(k, ct.float32)
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
    q = ct.astype(q, ct.float32)
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
        k = ct.astype(k, ct.float32)

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

    for j in range((SEQ + BLOCK_N - 1) // BLOCK_N ):
        k = ct.load(K, # loading from tensor K
            (batch_idx, head_idx, j, 0), # tile index
            (1, 1, BLOCK_N, HEAD_DIM), # how much to load (1 batch, 1 head, BLOCK_N rows/SEQ, full HEAD_DIM)
            latency=2) 

        v = ct.load(V, # loading from tensor V
            (batch_idx, head_idx, j, 0), # tile index
            (1, 1, BLOCK_N, HEAD_DIM), # how much to load (1 batch, 1 head, BLOCK_N rows/SEQ, full HEAD_DIM)
            latency=4)

        # squeeze tensors (to remove batch and head dims)
        k = ct.reshape(k, (BLOCK_N, HEAD_DIM))
        k_t = ct.transpose(k)  
        # squeeze tensors (to remove batch and head dims)
        v = ct.reshape(v, ((BLOCK_N, HEAD_DIM)))

    
        # matmul
        acc = ct.zeros((BLOCK_M, BLOCK_N), dtype=ct.float32)
        acc = ct.mma(q,k_t, acc) # (BLOCK_M, HEAD_DIM) @ (HEAD_DIM, BLOCK_N) -> (BLOCK_M, BLOCK_N)


        # online softmax: maximum of tile
        m_tile = ct.max(acc, axis=-1, keepdims=True) * qk_scale
        m_new = ct.maximum(m, m_tile)
        
        # correction factor: rescale previous results to use new max
        correction = ct.exp2(m - m_new, flush_to_zero=True)
        l = correction * l # l originally contains sum(exp(s - m_old)), correction rescales to sum(exp(s - m_new))
        o = correction * o # o originally contains o = exp(s0 - m_old) @ V0, so after correction -> exp(s0 - m_new) @ V0 = exp(s0 - m_old) * exp(m_old - m_new) @ V0: 

        # accumulate this tile's contribution
        softmax_numerator_tile = ct.exp2(acc  * qk_scale - m_new, flush_to_zero=True)
        l = l + ct.sum(softmax_numerator_tile, axis=-1, keepdims=True)

        o = ct.mma(softmax_numerator_tile.astype(ct.float16), v, o) 

        m = m_new

    # normalize by full row sum
    o = ct.truediv(o, l, flush_to_zero=True, rounding_mode=RMd.APPROX)

    ct.store(O, # output mem
             (batch_idx, head_idx, query_row_idx, 0), # tile index
             ct.reshape(ct.astype(o, ct.float16), (1, 1, BLOCK_M, HEAD_DIM)) # cutile array (unsqueezed) and cast back to float16
             )
def cutile_flash_attention_v2(Q, K, V):
    batch, num_heads, seq, head_dim = Q.shape
    stream = torch.cuda.current_stream().cuda_stream
    grid = (batch, num_heads, math.ceil(seq/BLOCK_M))
    O = torch.empty_like(Q)
    INV_LOG_2 = 1.0 / math.log(2)
    qk_scale = (1.0 / math.sqrt(head_dim)) * INV_LOG_2
    ct.launch(stream, grid, cutile_flash_attention_v2_kernel, (Q, K, V, qk_scale, seq, head_dim, O))
    return O


def run_benchmarks():
    configs = [
        (32, 128, 128, 32),     # small
        (4, 32, 512, 64),       # medium
        (4, 32, 2048, 64),      # large
    ]

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
            "PyTorch (optimized)": pytorch_attention,
            "PyTorch (manual)": pytorch_manual_attention,
            "Flash v1": cutile_flash_attention_v1,
            "Flash v2": cutile_flash_attention_v2,
        }
        # only include naive kernels for small SEQ (v1 hangs on large)
        if SEQ <= 128:
            fns["Cutile v1 (naive)"] = cutile_attention_v1
        if SEQ <= 2048:
            fns["Cutile v2 (naive)"] = cutile_attention_v2

        results = benchmark(fns, Q, K, V, ref_fn=pytorch_attention)
        all_results.append(results)
        config_labels.append(f"B={BATCH} H={NUM_HEADS}\nSEQ={SEQ} D={HEAD_DIM}")

    plot_benchmarks(all_results, config_labels)

if __name__=="__main__":
    run_benchmarks()