import torch
import triton
import triton.language as tl

@triton.jit
def evict_fwd_kernel(
    # Inputs
    Q_ptr, K_ptr, V_ptr, Mask_ptr,
    # Outputs
    AttnOutput_ptr, TempLoss_ptr, L_ptr,
    # Parameters
    scaling,
    N_REP,
    N_Q_HEADS, # Total number of query heads
    # Strides
    stride_q_b, stride_q_h, stride_q_d,
    stride_k_b, stride_k_h_kv, stride_k_s, stride_k_d,
    stride_v_b, stride_v_h_kv, stride_v_s, stride_v_d,
    stride_mask_b, stride_mask_s,
    stride_ao_b, stride_ao_h, stride_ao_d,
    stride_loss_b, stride_loss_h, stride_loss_s,
    stride_l_b, stride_l_h, stride_l_s,
    # Compile-time constants for tensor dimensions
    D_HEAD: tl.constexpr,
    SEQ_LEN_KV: tl.constexpr,
    BLOCK_S: tl.constexpr, # New: Block size for SEQ_LEN_KV dimension
):
    # Program IDs for batch and query head
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    # --- Common offsets for D_HEAD dimension ---
    d_offs = tl.arange(0, D_HEAD)

    # --- Load Query vector ---
    # Q is (B, N_Q_H, 1, D_HEAD). We only care about the single sequence element.
    q_base_ptr = Q_ptr + pid_b * stride_q_b + pid_h * stride_q_h
    q_vec = tl.load(q_base_ptr + d_offs * stride_q_d) # Shape: (D_HEAD,)

    # --- Determine Key/Value head index ---
    kv_head_idx = pid_h // N_REP

    # --- Base pointers for K, V, Mask for the current batch item and KV head ---
    k_batch_head_base_ptr = K_ptr + pid_b * stride_k_b + kv_head_idx * stride_k_h_kv
    v_batch_head_base_ptr = V_ptr + pid_b * stride_v_b + kv_head_idx * stride_v_h_kv
    mask_batch_base_ptr = Mask_ptr + pid_b * stride_mask_b
    loss_base_ptr = TempLoss_ptr + pid_b * stride_loss_b + pid_h * stride_loss_h

    # --- Accumulators for Online Softmax and Attention Output ---
    # Accumulator for the output, initialized to zeros. Type: float32 for precision.
    acc_o = tl.zeros((D_HEAD,), dtype=tl.float32)
    l_i = tl.zeros((1,), dtype=tl.float32)

    # === PASS 1: Compute Attention Output using Online Softmax ===
    for s_start_offset in range(0, SEQ_LEN_KV, BLOCK_S):
        s_block_offs = s_start_offset + tl.arange(0, BLOCK_S)
        s_boundary_mask = s_block_offs < SEQ_LEN_KV
        k_s_ptrs = k_batch_head_base_ptr + s_block_offs[:, None] * stride_k_s
        k_block_ptrs = k_s_ptrs + d_offs[None, :] * stride_k_d
        k_block = tl.load(k_block_ptrs, mask=s_boundary_mask[:, None], other=0.0)
        scores_block = tl.sum(q_vec[None, :].to(tl.float32) * k_block.to(tl.float32), axis=1) * scaling
        
        mask_s_ptrs = mask_batch_base_ptr + s_block_offs * stride_mask_s
        attn_mask_block_vec = tl.load(mask_s_ptrs, mask=s_boundary_mask, other=False) 
        scores_block = tl.where(attn_mask_block_vec & s_boundary_mask, scores_block, float(-1e35))
        
        p_block_numerators = tl.exp(scores_block)
        l_i += tl.sum(p_block_numerators, axis=0)
        
        v_s_ptrs = v_batch_head_base_ptr + s_block_offs[:, None] * stride_v_s
        v_block_ptrs = v_s_ptrs + d_offs[None, :] * stride_v_d
        v_block = tl.load(v_block_ptrs, mask=s_boundary_mask[:, None], other=0.0)
        
        attn_output_tmp = p_block_numerators[:, None] * v_block.to(tl.float32)
        acc_o += tl.sum(attn_output_tmp, axis=0)
        attn_output_loss = tl.sum(tl.abs(attn_output_tmp), axis=1)/D_HEAD
        tl.store(loss_base_ptr + s_block_offs * stride_loss_s, attn_output_loss, mask = s_boundary_mask)

    attn_output_vec = (acc_o / l_i).to(Q_ptr.dtype.element_ty)

    ao_base_ptr = AttnOutput_ptr + pid_b * stride_ao_b + pid_h * stride_ao_h
    tl.store(ao_base_ptr + d_offs * stride_ao_d, attn_output_vec)
    
    tl.store(L_ptr + pid_b * stride_l_b + pid_h * stride_l_h + tl.arange(0, 1) * stride_l_s, l_i)



def triton_evict_attention_forward(
    query: torch.Tensor,        # [B, N_Q_HEADS, 1, D_HEAD]
    key: torch.Tensor,          # [B, N_KV_HEADS, S_KV, D_HEAD]
    value: torch.Tensor,        # [B, N_KV_HEADS, S_KV, D_HEAD]
    attention_mask: torch.Tensor, # [B, S_KV], dtype=torch.bool (True means keep)
    num_key_value_groups: int,  # This is N_REP
    scaling: float,
    dropout: float = 0.0,
    block_s: int = 64,          # Block size for SEQ_LEN_KV, can be tuned
):
    assert query.is_cuda and key.is_cuda and value.is_cuda and attention_mask.is_cuda
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    attention_mask = attention_mask.contiguous()

    if dropout != 0.0:
        print("Warning: Dropout is not implemented in the Triton kernel for evict_attention_forward.")

    BATCH_SIZE, N_Q_HEADS, _, D_HEAD = query.shape
    _, N_KV_HEADS, SEQ_LEN_KV, _ = key.shape

    if N_Q_HEADS // num_key_value_groups != N_KV_HEADS:
         raise ValueError(
            f"N_Q_HEADS ({N_Q_HEADS}) // num_key_value_groups ({num_key_value_groups}) "
            f"must be equal to N_KV_HEADS ({N_KV_HEADS})"
        )

    attn_output = torch.empty_like(query)
    temp_loss = torch.zeros((BATCH_SIZE, N_Q_HEADS, SEQ_LEN_KV), dtype=torch.float32, device=query.device)
    tmp_l = torch.zeros((BATCH_SIZE, N_Q_HEADS, 1), dtype=torch.float32, device=query.device)

    stride_q_b, stride_q_h, _, stride_q_d = query.stride()
    stride_k_b, stride_k_h_kv, stride_k_s, stride_k_d = key.stride()
    stride_v_b, stride_v_h_kv, stride_v_s, stride_v_d = value.stride()
    stride_mask_b, stride_mask_s = attention_mask.stride()
    stride_ao_b, stride_ao_h, _, stride_ao_d = attn_output.stride()
    stride_loss_b, stride_loss_h, stride_loss_s = temp_loss.stride()
    stride_l_b, stride_l_h, stride_l_s = tmp_l.stride()

    grid = (BATCH_SIZE, N_Q_HEADS)
    
    # Determine a reasonable number of warps.
    # More warps can help hide latency, but too many can lead to register spilling or occupancy issues.
    # 4 or 8 are common choices.
    num_warps = 4
    if D_HEAD >= 128: # For larger head dimensions, more warps might be beneficial
        num_warps = 8


    evict_fwd_kernel[grid](
        Q_ptr=query, K_ptr=key, V_ptr=value, Mask_ptr=attention_mask,
        AttnOutput_ptr=attn_output, TempLoss_ptr=temp_loss, L_ptr=tmp_l,
        scaling=scaling,
        N_REP=num_key_value_groups,
        N_Q_HEADS=N_Q_HEADS,
        stride_q_b=stride_q_b, stride_q_h=stride_q_h, stride_q_d=stride_q_d,
        stride_k_b=stride_k_b, stride_k_h_kv=stride_k_h_kv, stride_k_s=stride_k_s, stride_k_d=stride_k_d,
        stride_v_b=stride_v_b, stride_v_h_kv=stride_v_h_kv, stride_v_s=stride_v_s, stride_v_d=stride_v_d,
        stride_mask_b=stride_mask_b, stride_mask_s=stride_mask_s,
        stride_ao_b=stride_ao_b, stride_ao_h=stride_ao_h, stride_ao_d=stride_ao_d,
        stride_loss_b=stride_loss_b, stride_loss_h=stride_loss_h, stride_loss_s=stride_loss_s,
        stride_l_b=stride_l_b, stride_l_h=stride_l_h, stride_l_s=stride_l_s,
        D_HEAD=D_HEAD,
        SEQ_LEN_KV=SEQ_LEN_KV,
        BLOCK_S=block_s, # Pass the new block size
        num_warps=num_warps
    )
    final_loss = (temp_loss / tmp_l).mean(dim=1).to(query.dtype)

    attn_output_final = attn_output.transpose(1, 2).contiguous()
    evict_idx = torch.argmin(final_loss, dim=-1)

    return attn_output_final, evict_idx