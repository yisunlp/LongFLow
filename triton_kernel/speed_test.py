import torch
import torch.nn as nn
import time
import math

# Attempt to import the user's Triton-based attention
# This line needs to work in your environment where 'attention.py' or the 'attention' module is accessible.
try:
    from attention import triton_evict_attention_forward
    TRITON_AVAILABLE = True
    print("Successfully imported triton_evict_attention_forward from attention module.")
except ImportError:
    TRITON_AVAILABLE = False
    print("Failed to import triton_evict_attention_forward from attention module. Triton version will not be benchmarked.")
    # Define a placeholder if not available, so the script structure holds
    def triton_evict_attention_forward(*args, **kwargs):
        raise NotImplementedError("Triton version could not be imported and is not available for benchmarking.")

# Provided repeat_kv function
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

# User's updated evict_attention_forward function
def evict_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor, # Expected shape [bs, kv_seq_len], dtype=torch.bool
    num_key_value_groups: int,
    scaling: float,
    dropout: float = 0.0 # dropout is not used in the function body provided
):
    # query: [bs, num_attention_heads, query_seq_len, head_dim]
    # key:   [bs, num_key_value_heads, kv_seq_len, head_dim]
    # value: [bs, num_key_value_heads, kv_seq_len, head_dim]
    # attention_mask: [bs, kv_seq_len] (bool)

    # Note on the condition:
    # attention_mask.sum() sums all True values (1s) in the mask.
    # If bs > 1, and all masks are True, attention_mask.sum().item() would be bs * key.shape[2].
    # So, key.shape[2] (kv_seq_len) would only equal this sum if bs=1 and all mask elements are True.
    # For bs > 1, this condition (attention_mask.sum().item() == key.shape[2]) will likely be False
    # even if all masks are effectively "no-op" (all True).
    # The 'else' branch will be taken for bs > 1 or if any mask element is False when bs=1.

    if attention_mask.sum().item() == key.shape[2]: # Checks if total number of True elements in mask equals kv_seq_len
        # This branch is taken if bs=1 and mask is all True, or if mask sums up to kv_seq_len (e.g. partial masks in a batch summing up, unlikely intent)
        key_states = repeat_kv(key, num_key_value_groups)
        value_states = repeat_kv(value, num_key_value_groups)
        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        # importance = attn_weights.squeeze(2).unsqueeze(-1)*value_states # Original
        # Corrected importance/attn_output logic path for standard attention:
        # attn_output = torch.matmul(attn_weights, value_states)
        # The user's specific importance calculation:
        importance = attn_weights.squeeze(2).unsqueeze(-1) * value_states # [bs, num_attn_heads, kv_seq_len, head_dim]
        attn_output = importance.sum(dim=2, keepdim=True) # [bs, num_attn_heads, 1, head_dim]
        loss = (importance.abs()).mean(dim=(1, -1)) # [bs, kv_seq_len]
        attn_output = attn_output.transpose(1, 2).contiguous() # [bs, 1, num_attn_heads, head_dim]
        evict_idx = torch.argmin(loss, dim=-1) # [bs]
    else:
        key_states = repeat_kv(key, num_key_value_groups)
        value_states = repeat_kv(value, num_key_value_groups)
        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        min_dtype = torch.finfo(attn_weights.dtype).min
        # attention_mask: [bs, kv_seq_len] -> unsqueeze(1).unsqueeze(2) -> [bs, 1, 1, kv_seq_len]
        # This mask is broadcastable to attn_weights [bs, num_attention_heads, query_seq_len, kv_seq_len]
        attn_weights = attn_weights.masked_fill(~attention_mask.unsqueeze(1).unsqueeze(2), min_dtype)
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)

        # importance = attn_weights.squeeze(2).unsqueeze(-1)*value_states # Original
        # Corrected importance/attn_output logic path for standard attention:
        # attn_output = torch.matmul(attn_weights, value_states)
        # The user's specific importance calculation:
        importance = attn_weights.squeeze(2).unsqueeze(-1) * value_states
        attn_output = importance.sum(dim=2, keepdim=True)
        loss = (importance.abs()).mean(dim=(1, -1))
        attn_output = attn_output.transpose(1, 2).contiguous()
        evict_idx = torch.argmin(loss, dim=-1)
    return attn_output, evict_idx


def benchmark(batch_sizes, num_k_v_groups, device_str='cuda' if torch.cuda.is_available() else 'cpu'):
    print(f"Benchmarking on {device_str}")

    # Attention parameters
    num_attention_heads = 32
    num_key_value_heads = 8 # num_k_v_groups = num_attention_heads / num_key_value_heads
    query_seq_len = 1
    kv_seq_len = 1024
    head_dim = 128
    dtype = torch.float16
    scaling = 1.0 / math.sqrt(head_dim)

    results = {}
    
    # Two mask scenarios
    mask_scenarios = {
        "all_true_mask": torch.ones((1, kv_seq_len), dtype=torch.bool, device=device_str), # Will be expanded to batch size
        "mixed_mask": (torch.rand((1, kv_seq_len), device=device_str) > 0.1).bool() # Approx 90% True
    }


    for mask_name, base_mask_pattern in mask_scenarios.items():
        print(f"\n===== Mask Scenario: {mask_name} =====")
        results[mask_name] = {}
        for bs in batch_sizes:
            print(f"\n--- Batch Size: {bs} ---")
            q_shape = (bs, num_attention_heads, query_seq_len, head_dim)
            k_shape = (bs, num_key_value_heads, kv_seq_len, head_dim)
            v_shape = (bs, num_key_value_heads, kv_seq_len, head_dim)
            
            # Generate random data
            query = torch.randn(q_shape, dtype=dtype, device=device_str)
            key = torch.randn(k_shape, dtype=dtype, device=device_str)
            value = torch.randn(v_shape, dtype=dtype, device=device_str)
            
            # Prepare attention mask for the current batch size
            attention_mask = base_mask_pattern.expand(bs, -1).clone() # [bs, kv_seq_len]

            # Warm-up runs
            for _ in range(5):
                _, _ = evict_attention_forward(query, key, value, attention_mask, num_k_v_groups, scaling)
                if TRITON_AVAILABLE:
                    try:
                        _, _ = triton_evict_attention_forward(query, key, value, attention_mask, num_k_v_groups, scaling)
                    except NotImplementedError:
                        pass # Handled by TRITON_AVAILABLE check mostly
                    except Exception as e:
                        print(f"Warmup error with Triton: {e}")
                        # Potentially disable Triton for subsequent runs if it keeps failing
            if device_str == 'cuda':
                torch.cuda.synchronize()

            # Benchmark PyTorch version
            start_time = time.time()
            for _ in range(20): # Number of iterations for timing
                _, _ = evict_attention_forward(query, key, value, attention_mask, num_k_v_groups, scaling)
            if device_str == 'cuda':
                torch.cuda.synchronize()
            end_time = time.time()
            pytorch_time = (end_time - start_time) / 20
            print(f"PyTorch evict_attention_forward: {pytorch_time:.6f} seconds per call")
            results[mask_name].setdefault('pytorch', []).append(pytorch_time)

            # Benchmark Triton version
            if TRITON_AVAILABLE:
                try:
                    start_time = time.time()
                    for _ in range(20): # Number of iterations for timing
                         # Ensure triton_evict_attention_forward has the same signature or adapt the call
                        _, _ = triton_evict_attention_forward(query, key, value, attention_mask, num_k_v_groups, scaling)
                    if device_str == 'cuda':
                        torch.cuda.synchronize()
                    end_time = time.time()
                    triton_time = (end_time - start_time) / 20
                    print(f"Triton evict_attention_forward:  {triton_time:.6f} seconds per call")
                    results[mask_name].setdefault('triton', []).append(triton_time)
                except NotImplementedError:
                    print("Triton evict_attention_forward: Not implemented or import failed, skipping.")
                except Exception as e:
                    print(f"Error running Triton version: {e}. Skipping for this configuration.")
            else:
                print("Triton evict_attention_forward: Not available, skipping.")


    print("\n\n--- Overall Summary ---")
    for mask_name, mask_results in results.items():
        print(f"\n===== Results for Mask Scenario: {mask_name} =====")
        for bs_idx, bs in enumerate(batch_sizes):
            print(f"Batch Size: {bs}")
            if 'pytorch' in mask_results and len(mask_results['pytorch']) > bs_idx:
                pytorch_t = mask_results['pytorch'][bs_idx]
                print(f"  PyTorch: {pytorch_t:.6f} s")
                if TRITON_AVAILABLE and 'triton' in mask_results and len(mask_results['triton']) > bs_idx:
                    triton_t = mask_results['triton'][bs_idx]
                    print(f"  Triton:  {triton_t:.6f} s")
                    if triton_t > 0 : # Avoid division by zero
                        speedup = pytorch_t / triton_t
                        print(f"  Speedup (Triton vs PyTorch): {speedup:.2f}x")
                elif 'triton' not in mask_results or len(mask_results['triton']) <= bs_idx :
                     print(f"  Triton:  Not benchmarked or error occurred.")


if __name__ == "__main__":
    batch_sizes_to_test = [1, 2, 4, 8, 16, 32,64,128]
    num_kv_groups_param = 4 # num_attention_heads (32) / num_key_value_heads (8)
    
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, running on CPU. Performance will be significantly different, especially for fp16 and Triton.")
        benchmark(batch_sizes_to_test, num_kv_groups_param, device_str='cpu')
    else:
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        benchmark(batch_sizes_to_test, num_kv_groups_param, device_str='cuda')