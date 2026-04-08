import cuda.tile as ct
import torch
import torch.nn.functional as F
import math
from utils import time_fn, compute_error

BATCH = 32
NUM_HEADS = 128 
SEQ = 64
HEAD_DIM = 32 # need to be a power of 2
BLOCK_M = 64 # tiling the SEQ into tiles of 64

Q = torch.randn(BATCH, NUM_HEADS, SEQ, HEAD_DIM, dtype=torch.float16, device="cuda")
K = torch.randn(BATCH, NUM_HEADS, SEQ, HEAD_DIM, dtype=torch.float16, device="cuda")
V = torch.randn(BATCH, NUM_HEADS, SEQ, HEAD_DIM, dtype=torch.float16, device="cuda")


def pytorch_manual_attention(Q, K, V):
    # attention(q,k,v) = softmax(Q @ K^T / sqrt(d_k)) @ V
    # d_k -> dimension of the key vectors (HEAD_DIM in this code)
    # softmax x_i = e^x_i / sum (j = 1->N) (e^x_j)
    return F.softmax(( Q @  K.transpose(-2, -1) )/ math.sqrt(HEAD_DIM), dim=-1) @ V

def pytorch_attention(Q, K, V):
    return F.scaled_dot_product_attention(Q, K, V)


@ct.kernel
def cutile_attention_v1_kernel(Q: ct.Array, 
                               K: ct.Array, 
                               V: ct.Array,
                               scale: float,
                               O: ct.Array # output,
                               ):
    batch_idx = ct.bid(0)
    head_idx = ct.bid(1)
    query_row_idx = ct.bid(2)

    q = ct.load(Q, # loading from tensor Q
                (batch_idx, head_idx, query_row_idx * BLOCK_M, 0), # starting index for each dimension
                (1, 1, BLOCK_M, HEAD_DIM) # how much to load (1 batch, 1 head, BLOCK_M rows/SEQ, full HEAD_DIM)
                )
    q = ct.astype(q, ct.float32)

    k = ct.load(K, # loading from tensor K
            (batch_idx, head_idx, 0, 0), # starting index for each dimension
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
            (batch_idx, head_idx, 0, 0), # starting index for each dimension
            (1, 1, SEQ, HEAD_DIM) # how much to load (1 batch, 1 head, full SEQ, full HEAD_DIM)
            ) 
    v = ct.reshape(v, ((SEQ, HEAD_DIM)))
    v = ct.astype(v, ct.float32)
    
    out = ct.zeros((BLOCK_M, HEAD_DIM), dtype=ct.float32)
    out = ct.mma(acc, v, out) # (BLOCK_M, SEQ) @ (SEQ, HEAD_DIM) -> (BLOCK_M, HEAD_DIM)

    ct.store(O, # output mem
             (batch_idx, head_idx, query_row_idx * BLOCK_M, 0), # starting index for each dimension
             ct.reshape(ct.astype(out, ct.float16), (1, 1, BLOCK_M, HEAD_DIM)) # cutile array (unsqueezed) and cast back to float16
             )
    

def cutile_attention_v1(Q, K, V):
    stream = torch.cuda.current_stream().cuda_stream                                                                                                       
    grid = (BATCH, NUM_HEADS, math.ceil(SEQ/BLOCK_M))            
    O = torch.empty_like(Q)
    scale = 1.0 / math.sqrt(HEAD_DIM)                                                                              
    ct.launch(stream, grid, cutile_attention_v1_kernel, (Q, K, V, scale, O))   
    return O

print(compute_error(cutile_attention_v1(Q, K, V), pytorch_attention(Q, K, V)))
print(compute_error(pytorch_manual_attention(Q, K, V), pytorch_attention(Q, K, V)))

print(pytorch_attention(Q,K,V).shape)
print(pytorch_manual_attention(Q,K,V).shape)
print(cutile_attention_v1(Q,K,V).shape)

print(time_fn(pytorch_attention, Q,K,V))
print(time_fn(pytorch_manual_attention, Q,K,V))
print(time_fn(cutile_attention_v1, Q,K,V))