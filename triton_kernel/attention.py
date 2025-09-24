import torch
import triton
import triton.language as tl

@triton.jit
def evict_fwd_kernel(
    # Inputs
    Q_ptr, K_ptr, V_ptr, Mask_ptr,
    # Outputs
    AttnOutput_ptr, TempLoss_ptr,
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
    stride_loss_b, stride_loss_s,
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

    # --- Accumulators for Online Softmax and Attention Output ---
    # Accumulator for the output, initialized to zeros. Type: float32 for precision.
    acc_o = tl.zeros((D_HEAD,), dtype=tl.float32)
    # Max score accumulator for softmax, initialized to -infinity. Type: float32.
    m_i = tl.full((1,), float(-1e35), dtype=tl.float32)
    # Sum of exponentials accumulator for softmax, initialized to zeros. Type: float32.
    l_i = tl.zeros((1,), dtype=tl.float32)

    # === PASS 1: Compute Attention Output using Online Softmax ===
    for s_start_offset in range(0, SEQ_LEN_KV, BLOCK_S):
        # --- Define sequence offsets for the current block ---
        s_block_offs = s_start_offset + tl.arange(0, BLOCK_S)
        # Mask for elements within the actual sequence length
        s_boundary_mask = s_block_offs < SEQ_LEN_KV

        # --- Load K block ---
        # Pointers for K block: (BLOCK_S, D_HEAD)
        k_s_ptrs = k_batch_head_base_ptr + s_block_offs[:, None] * stride_k_s
        k_block_ptrs = k_s_ptrs + d_offs[None, :] * stride_k_d
        # Load K block, masking out-of-bounds elements
        k_block = tl.load(k_block_ptrs, mask=s_boundary_mask[:, None], other=0.0) # Shape: (BLOCK_S, D_HEAD)

        # --- Compute Attention Scores for the block (Q @ K_block^T) ---
        # Convert query and key to float32 for score computation to maintain precision.
        scores_block = tl.sum(q_vec[None, :].to(tl.float32) * k_block.to(tl.float32), axis=1) * scaling # Shape: (BLOCK_S,)

        # --- Apply Attention Mask for the block ---
        mask_s_ptrs = mask_batch_base_ptr + s_block_offs * stride_mask_s
        # Load mask block, masking out-of-bounds elements
        attn_mask_block_vec = tl.load(mask_s_ptrs, mask=s_boundary_mask, other=False) # Shape: (BLOCK_S,)
        # Apply mask: where mask is False or outside sequence boundary, set score to -infinity.
        scores_block = tl.where(attn_mask_block_vec & s_boundary_mask, scores_block, float(-1e35))

        # --- Online Softmax: Update m_i (max_score), l_i (sum_exp), and acc_o (output accumulator) ---
        m_i_prev = m_i
        m_block_max = tl.max(scores_block, axis=0)  # Max score in this block (scalar)
        m_i_new = tl.maximum(m_i_prev, m_block_max) # Update overall max score

        # Calculate numerators for softmax: p_block = exp(scores_block - m_i_new)
        p_block_numerators = tl.exp(scores_block - m_i_new) # Shape: (BLOCK_S,)

        # Rescale l_i and acc_o using the change in max score to maintain numerical stability
        exp_m_diff = tl.exp(m_i_prev - m_i_new) # Scalar
        l_i = l_i * exp_m_diff
        acc_o = acc_o * exp_m_diff # Broadcasting scalar to (D_HEAD,)

        # Update l_i with sum of current block's numerators
        l_i += tl.sum(p_block_numerators, axis=0) # Sum over BLOCK_S

        # --- Load V block ---
        v_s_ptrs = v_batch_head_base_ptr + s_block_offs[:, None] * stride_v_s
        v_block_ptrs = v_s_ptrs + d_offs[None, :] * stride_v_d
        # Load V block, masking out-of-bounds elements
        v_block = tl.load(v_block_ptrs, mask=s_boundary_mask[:, None], other=0.0) # Shape: (BLOCK_S, D_HEAD)

        # --- Update Attention Output Accumulator (acc_o) ---
        # acc_o += P_block_numerators @ V_block (element-wise then sum)
        # Convert V to float32 for accumulation if it's not already.
        current_block_o = tl.sum(p_block_numerators[:, None] * v_block.to(tl.float32), axis=0) # Shape: (D_HEAD,)
        acc_o += current_block_o

        m_i = m_i_new # Store the updated max score for the next iteration

    # --- Finalize Attention Output ---
    # attn_output_vec = acc_o / l_i. Cast to query's dtype.
    # Handle l_i == 0 case (e.g., if all scores were -inf due to masking) to avoid division by zero.
    l_i_safe = tl.where(l_i == 0, 1.0, l_i) # If l_i is 0, acc_o is also 0, so 0/1 = 0.
    attn_output_vec = (acc_o / l_i_safe).to(Q_ptr.dtype.element_ty)

    # --- Store Attention Output ---
    ao_base_ptr = AttnOutput_ptr + pid_b * stride_ao_b + pid_h * stride_ao_h
    tl.store(ao_base_ptr + d_offs * stride_ao_d, attn_output_vec)

    # === PASS 2: Calculate Loss Contribution for TempLoss ===
    # This pass uses the final m_i and l_i_safe computed in Pass 1.
    loss_batch_base_ptr = TempLoss_ptr + pid_b * stride_loss_b
    for s_start_offset in range(0, SEQ_LEN_KV, BLOCK_S):
        s_block_offs = s_start_offset + tl.arange(0, BLOCK_S)
        s_boundary_mask = s_block_offs < SEQ_LEN_KV

        # --- Reload K block and re-compute scores_block (as in Pass 1) ---
        k_s_ptrs = k_batch_head_base_ptr + s_block_offs[:, None] * stride_k_s
        k_block_ptrs = k_s_ptrs + d_offs[None, :] * stride_k_d
        k_block = tl.load(k_block_ptrs, mask=s_boundary_mask[:, None], other=0.0)
        scores_block = tl.sum(q_vec[None, :].to(tl.float32) * k_block.to(tl.float32), axis=1) * scaling

        # --- Apply Attention Mask for the block (as in Pass 1) ---
        mask_s_ptrs = mask_batch_base_ptr + s_block_offs * stride_mask_s
        attn_mask_block_vec = tl.load(mask_s_ptrs, mask=s_boundary_mask, other=False)
        scores_block = tl.where(attn_mask_block_vec & s_boundary_mask, scores_block, float("-inf"))

        # --- Calculate final attention probabilities for this block ---
        # Use final m_i and l_i_safe from Pass 1 for normalization.
        attn_probs_block_f32 = tl.exp(scores_block - m_i) / l_i_safe # m_i is the final max_score
        attn_probs_block = attn_probs_block_f32.to(Q_ptr.dtype.element_ty) # Cast to query's dtype

        # --- Reload V block (as in Pass 1) ---
        v_s_ptrs = v_batch_head_base_ptr + s_block_offs[:, None] * stride_v_s
        v_block_ptrs = v_s_ptrs + d_offs[None, :] * stride_v_d
        v_block = tl.load(v_block_ptrs, mask=s_boundary_mask[:, None], other=0.0)

        # --- Calculate Importance for this block ---
        # importance_matrix_block has shape (BLOCK_S, D_HEAD)
        importance_matrix_block = attn_probs_block[:, None] * v_block

        # --- Sum absolute importance over D_HEAD dimension for this block ---
        # Accumulate in float32 for precision before casting for atomic_add.
        abs_importance_block_f32 = tl.abs(importance_matrix_block.to(tl.float32))
        sum_abs_importance_over_d_block_f32 = tl.sum(abs_importance_block_f32, axis=1) # Shape: (BLOCK_S,)
        
        # Cast to the dtype of TempLoss before atomic operation
        sum_abs_importance_over_d_block_final_dtype = sum_abs_importance_over_d_block_f32.to(TempLoss_ptr.dtype.element_ty)

        # --- Atomically add this block's contribution to TempLoss ---
        loss_s_indices_block = s_start_offset + tl.arange(0, BLOCK_S)
        current_loss_value_ptrs = loss_batch_base_ptr + loss_s_indices_block * stride_loss_s
        # Perform atomic add only for valid sequence elements within the block.
        tl.atomic_add(current_loss_value_ptrs, sum_abs_importance_over_d_block_final_dtype, mask=s_boundary_mask)


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
    temp_loss = torch.zeros((BATCH_SIZE, SEQ_LEN_KV), dtype=query.dtype, device=query.device)

    stride_q_b, stride_q_h, _, stride_q_d = query.stride()
    stride_k_b, stride_k_h_kv, stride_k_s, stride_k_d = key.stride()
    stride_v_b, stride_v_h_kv, stride_v_s, stride_v_d = value.stride()
    stride_mask_b, stride_mask_s = attention_mask.stride()
    stride_ao_b, stride_ao_h, _, stride_ao_d = attn_output.stride()
    stride_loss_b, stride_loss_s = temp_loss.stride()

    grid = (BATCH_SIZE, N_Q_HEADS)
    
    # Determine a reasonable number of warps.
    # More warps can help hide latency, but too many can lead to register spilling or occupancy issues.
    # 4 or 8 are common choices.
    num_warps = 4
    if D_HEAD >= 128: # For larger head dimensions, more warps might be beneficial
        num_warps = 8


    evict_fwd_kernel[grid](
        Q_ptr=query, K_ptr=key, V_ptr=value, Mask_ptr=attention_mask,
        AttnOutput_ptr=attn_output, TempLoss_ptr=temp_loss,
        scaling=scaling,
        N_REP=num_key_value_groups,
        N_Q_HEADS=N_Q_HEADS,
        stride_q_b=stride_q_b, stride_q_h=stride_q_h, stride_q_d=stride_q_d,
        stride_k_b=stride_k_b, stride_k_h_kv=stride_k_h_kv, stride_k_s=stride_k_s, stride_k_d=stride_k_d,
        stride_v_b=stride_v_b, stride_v_h_kv=stride_v_h_kv, stride_v_s=stride_v_s, stride_v_d=stride_v_d,
        stride_mask_b=stride_mask_b, stride_mask_s=stride_mask_s,
        stride_ao_b=stride_ao_b, stride_ao_h=stride_ao_h, stride_ao_d=stride_ao_d,
        stride_loss_b=stride_loss_b, stride_loss_s=stride_loss_s,
        D_HEAD=D_HEAD,
        SEQ_LEN_KV=SEQ_LEN_KV,
        BLOCK_S=block_s, # Pass the new block size
        num_warps=num_warps
    )

    if N_Q_HEADS * D_HEAD > 0:
        final_loss = temp_loss / (N_Q_HEADS * D_HEAD)
    else:
        final_loss = torch.zeros_like(temp_loss)

    attn_output_final = attn_output.transpose(1, 2).contiguous()
    evict_idx = torch.argmin(final_loss, dim=-1)

    return attn_output_final, evict_idx