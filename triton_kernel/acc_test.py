import torch
import torch.nn as nn
import time

# Assuming your triton_evict_attention_forward is in attention.py
# from attention import triton_evict_attention_forward
# For this example, I'll mock the import if attention.py is not available
# and use the PyTorch function as a placeholder for the Triton one
# to make the script runnable standalone for structure demonstration.
# In your actual use, ensure the import 'from attention import triton_evict_attention_forward' works.
try:
    from attention import triton_evict_attention_forward as triton_evict_attention_actual
    print("Successfully imported 'triton_evict_attention_forward' from attention.py")
except ImportError:
    print("Could not import 'triton_evict_attention_forward' from attention.py.")
    print("Using PyTorch version as a placeholder for Triton function for testing structure.")
    # Placeholder: if triton kernel is not found, we'll use PyTorch for both
    # to allow the test structure to run.
    def triton_evict_attention_actual(query, key, value, attention_mask, num_key_value_groups, scaling, dropout=0.0):
        # This is a placeholder. In actual use, this will be your Triton kernel.
        # For demonstration, let's call the PyTorch version.
        print("--- Triton (Placeholder) Called ---")
        mock_module = MockModule(num_key_value_groups)
        # The PyTorch reference function needs the original key/value before repeat_kv if we follow its signature
        # The Triton function signature I provided expects original K, V and num_key_value_groups for internal handling (or direct use if already repeated)
        # Let's assume the triton_evict_attention_forward handles repeat_kv implicitly or takes already repeated K,V if num_key_value_groups=1
        # The signature of triton_evict_attention_forward in the prompt had num_key_value_groups as an argument.

        # The provided Triton wrapper in the previous turn expected original K,V.
        # So, we call it directly.
        # Note: The placeholder here won't show speed differences.
        # It's just to make the script runnable if attention.py is missing.
        # The PyTorch reference function `evict_attention_forward_pytorch_ref` will do its own repeat_kv.
        # For fair comparison, the triton function should also take the original K,V and num_key_value_groups.
        # The example call to triton_evict_attention_forward in the previous response expects original K, V.
        
        # For the sake of this placeholder, let's assume it mimics the PyTorch logic if the actual Triton isn't there
        # This is NOT a proper test of Triton if it falls back here.
        attn_output, evict_idx, loss = evict_attention_forward_pytorch_ref(
            MockModule(num_key_value_groups),
            query,
            key, # original key
            value, # original value
            attention_mask,
            scaling,
            dropout
        )
        return attn_output, evict_idx, loss # also returning loss for comparison

# PyTorch reference functions
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class MockModule(nn.Module):
    def __init__(self, n_kv_groups):
        super().__init__()
        self.num_key_value_groups = n_kv_groups

def evict_attention_forward_pytorch_ref( # Renamed to avoid conflict
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor, # Original key before repeat_kv
    value: torch.Tensor, # Original value before repeat_kv
    attention_mask: torch.Tensor, # Bool mask
    scaling: float,
    dropout: float = 0.0,
    **kwargs, # Added to match original signature if any other args are passed
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    min_dtype_val = torch.finfo(attn_weights.dtype).min
    attn_weights = attn_weights.masked_fill(~attention_mask.unsqueeze(1).unsqueeze(2), min_dtype_val)
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    importance = attn_weights.squeeze(2).unsqueeze(-1) * value_states
    attn_output_sum = importance.sum(dim=2, keepdim=True) # Renamed to avoid clash
    loss = (importance.abs()).mean(dim=(1, -1))
    attn_output_final = attn_output_sum.transpose(1, 2).contiguous() # Renamed to avoid clash
    evict_idx = torch.argmin(loss, dim=-1)
    return attn_output_final, evict_idx, loss # Return loss for comparison

def run_test_configuration(config, test_triton=True):
    batch_size = config["batch_size"]
    num_q_heads = config["num_q_heads"]
    num_kv_heads = config["num_kv_heads"]
    seq_len_kv = config["seq_len_kv"]
    head_dim = config["head_dim"]
    dtype = config["dtype"]
    test_name = config["name"]

    print(f"\n--- Testing: {test_name} ---")
    print(f"Shape: B={batch_size}, NQ_H={num_q_heads}, NKV_H={num_kv_heads}, S_KV={seq_len_kv}, D_H={head_dim}")
    print(f"DataType: {dtype}")

    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        print("Skipping bfloat16 test as it's not supported on this GPU.")
        return
    if dtype == torch.float16 and not torch.cuda.is_available(): # half precision CPU ops can be slow or unsupported
         pass # allow CPU fp16 for structure, but ideally CUDA

    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if not torch.cuda.is_available() and dtype != torch.float32:
            print(f"Warning: CUDA not available, {dtype} might be slow or have precision issues on CPU.")


        num_key_value_groups = num_q_heads // num_kv_heads
        if num_q_heads % num_kv_heads != 0:
            print(f"Skipping invalid configuration: num_q_heads ({num_q_heads}) must be divisible by num_kv_heads ({num_kv_heads}).")
            return

        query = torch.randn(batch_size, num_q_heads, 1, head_dim, device=device, dtype=dtype)
        key = torch.randn(batch_size, num_kv_heads, seq_len_kv, head_dim, device=device, dtype=dtype)
        value = torch.randn(batch_size, num_kv_heads, seq_len_kv, head_dim, device=device, dtype=dtype)
        attention_mask = torch.randint(0, 2, (batch_size, seq_len_kv), device=device, dtype=torch.bool)
        # Ensure some True values in mask for non-trivial softmax
        if not torch.any(attention_mask):
             attention_mask[0,0] = True


        scaling = 1.0 / (head_dim**0.5)
        mock_module = MockModule(num_key_value_groups)

        # PyTorch Reference
        print("Running PyTorch reference...")
        torch.cuda.synchronize() if device == 'cuda' else None
        start_time_pytorch = time.perf_counter()
        for _ in range(config.get("warmup_iters", 2)): # Warmup
            _, _, _ = evict_attention_forward_pytorch_ref(mock_module, query, key, value, attention_mask, scaling)
        torch.cuda.synchronize() if device == 'cuda' else None
        
        pytorch_run_times = []
        for _ in range(config.get("run_iters", 10)): # Timed runs
            torch.cuda.synchronize() if device == 'cuda' else None
            iter_start_time = time.perf_counter()
            attn_output_pytorch, evict_idx_pytorch, loss_pytorch = evict_attention_forward_pytorch_ref(
                mock_module, query, key, value, attention_mask, scaling
            )
            torch.cuda.synchronize() if device == 'cuda' else None
            pytorch_run_times.append(time.perf_counter() - iter_start_time)
        time_pytorch = sum(pytorch_run_times) / len(pytorch_run_times)
        print(f"PyTorch Avg Time: {time_pytorch * 1000:.3f} ms")

        if test_triton:
            print("Running Triton version...")
            # Assuming triton_evict_attention_actual is the imported Triton kernel wrapper
            torch.cuda.synchronize() if device == 'cuda' else None
            start_time_triton = time.perf_counter()
            for _ in range(config.get("warmup_iters", 2)): # Warmup
                 _, _, _ = triton_evict_attention_actual(query, key, value, attention_mask, num_key_value_groups, scaling)
            torch.cuda.synchronize() if device == 'cuda' else None

            triton_run_times = []
            for _ in range(config.get("run_iters", 10)): # Timed runs
                torch.cuda.synchronize() if device == 'cuda' else None
                iter_start_time = time.perf_counter()
                attn_output_triton, evict_idx_triton, loss_triton = triton_evict_attention_actual(
                    query, key, value, attention_mask, num_key_value_groups, scaling
                )
                torch.cuda.synchronize() if device == 'cuda' else None
                triton_run_times.append(time.perf_counter() - iter_start_time)
            time_triton = sum(triton_run_times) / len(triton_run_times)
            print(f"Triton Avg Time: {time_triton * 1000:.3f} ms")
            print(f"Speedup (Triton vs PyTorch): {time_pytorch / time_triton:.2f}x" if time_triton > 0 else "N/A (Triton time is zero)")

            # Correctness Comparison
            atol = 1e-3 if dtype != torch.float32 else 1e-6
            rtol = 1e-3 if dtype != torch.float32 else 1e-6

            print("\nCorrectness Checks:")
            if attn_output_triton.shape != attn_output_pytorch.shape:
                print(f"🔴 Attn Output shapes differ: Triton {attn_output_triton.shape}, PyTorch {attn_output_pytorch.shape}")
            elif torch.allclose(attn_output_triton, attn_output_pytorch, atol=atol, rtol=rtol):
                print("🟢 Attn Output: All Close")
            else:
                print("🔴 Attn Output: NOT All Close")
                # print("PyTorch output sample:", attn_output_pytorch.flatten()[:8])
                # print("Triton output sample:", attn_output_triton.flatten()[:8])
                # print("Difference sample:", (attn_output_pytorch - attn_output_triton).abs().flatten()[:8])
                print("Max Diff (Attn Output):", (attn_output_pytorch - attn_output_triton).abs().max())


            if loss_triton.shape != loss_pytorch.shape:
                 print(f"🔴 Loss shapes differ: Triton {loss_triton.shape}, PyTorch {loss_pytorch.shape}")
            elif torch.allclose(loss_triton, loss_pytorch, atol=atol, rtol=rtol):
                print("🟢 Loss: All Close")
            else:
                print("🔴 Loss: NOT All Close")
                # print("PyTorch loss sample:", loss_pytorch.flatten()[:8])
                # print("Triton loss sample:", loss_triton.flatten()[:8])
                # print("Difference sample:", (loss_pytorch - loss_triton).abs().flatten()[:8])
                print("Max Diff (Loss):", (loss_pytorch - loss_triton).abs().max())

            if evict_idx_triton.shape != evict_idx_pytorch.shape:
                print(f"🔴 Evict Idx shapes differ: Triton {evict_idx_triton.shape}, PyTorch {evict_idx_pytorch.shape}")
            elif torch.equal(evict_idx_triton, evict_idx_pytorch):
                print("🟢 Evict Idx: Equal")
            else:
                print("🟡 Evict Idx: NOT Equal (This can happen if multiple minimums exist in loss)")
                # Check if the loss values at the chosen indices are very close
                try:
                    loss_at_pytorch_idx = torch.gather(loss_pytorch, 1, evict_idx_pytorch.unsqueeze(1)).squeeze(1)
                    loss_at_triton_idx_from_pytorch_loss = torch.gather(loss_pytorch, 1, evict_idx_triton.unsqueeze(1)).squeeze(1)
                    loss_at_triton_idx_from_triton_loss = torch.gather(loss_triton, 1, evict_idx_triton.unsqueeze(1)).squeeze(1)

                    if torch.allclose(loss_at_pytorch_idx, loss_at_triton_idx_from_pytorch_loss, atol=atol*10, rtol=rtol*10): # Check if Triton's choice is also a minimum in PyTorch's loss
                        print("    INFO: Loss values at PyTorch's chosen index and Triton's chosen index (evaluated on PyTorch's loss) are close.")
                    else:
                        print("    INFO: Loss values at chosen indices differ significantly, or Triton's choice is not a minimum in PyTorch's loss.")
                    
                    # print(f"    PyTorch evict_idx: {evict_idx_pytorch[:4]}, Loss values: {loss_at_pytorch_idx[:4]}")
                    # print(f"    Triton  evict_idx: {evict_idx_triton[:4]}, Loss values (from PyTorch loss): {loss_at_triton_idx_from_pytorch_loss[:4]}")
                    # print(f"    Triton  evict_idx: {evict_idx_triton[:4]}, Loss values (from Triton loss): {loss_at_triton_idx_from_triton_loss[:4]}")

                except Exception as e:
                    print(f"    Error during evict_idx analysis: {e}")
        print("--- Test End ---")

    except Exception as e:
        print(f"🔴 ERROR during test '{test_name}': {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_configs = [
        {
            "name": "FP32 Small", "batch_size": 2, "num_q_heads": 4, "num_kv_heads": 2,
            "seq_len_kv": 32, "head_dim": 16, "dtype": torch.float32,
            "warmup_iters": 1, "run_iters": 5
        },
        {
            "name": "FP32 Medium", "batch_size": 4, "num_q_heads": 8, "num_kv_heads": 2,
            "seq_len_kv": 64, "head_dim": 32, "dtype": torch.float32,
            "warmup_iters": 2, "run_iters": 10
        },
        {
            "name": "FP16 Medium", "batch_size": 4, "num_q_heads": 8, "num_kv_heads": 4, # GQA (N_REP=2)
            "seq_len_kv": 128, "head_dim": 64, "dtype": torch.float16,
            "warmup_iters": 2, "run_iters": 10
        },
        {
            "name": "FP32 Large Head Dim", "batch_size": 2, "num_q_heads": 4, "num_kv_heads": 1, # MHA basically (N_REP=4)
            "seq_len_kv": 512, "head_dim": 128, "dtype": torch.float32,
            "warmup_iters": 2, "run_iters": 10
        },
         { # Test MQA (Multi-Query Attention where num_kv_heads=1)
            "name": "FP16 MQA", "batch_size": 2, "num_q_heads": 8, "num_kv_heads": 1,
            "seq_len_kv": 1024, "head_dim": 64, "dtype": torch.float16,
            "warmup_iters": 2, "run_iters": 10
        },
        { # Test with num_key_value_groups = 1 (num_q_heads == num_kv_heads)
            "name": "FP32 N_REP=1", "batch_size": 2, "num_q_heads": 4, "num_kv_heads": 4,
            "seq_len_kv": 64, "head_dim": 32, "dtype": torch.float32,
            "warmup_iters": 1, "run_iters": 5
        },
    ]

    # Check if the actual Triton kernel was imported to decide if we run full comparison
    try:
        _ = triton_evict_attention_actual # Check if it's defined
        # A more robust check would be to see if it's different from the placeholder
        is_triton_available = not ("Placeholder" in triton_evict_attention_actual.__doc__ if triton_evict_attention_actual.__doc__ else False)
        if "Placeholder" in triton_evict_attention_actual.__code__.co_filename : # Heuristic
             is_triton_available = False

    except NameError:
        is_triton_available = False

    if not is_triton_available:
        print("\nWARNING: Actual Triton kernel not found. Speed and correctness comparisons against Triton will use a PyTorch placeholder.")
        print("Ensure 'attention.py' contains your 'triton_evict_attention_forward' function for a real test.\n")


    for config in test_configs:
        run_test_configuration(config, test_triton=is_triton_available)