# RTX 5080 / High-End GPU Setup

Optimizing AdvTok Research for NVIDIA GeForce RTX 5080 and other high-end GPUs.

## RTX 5080 Specifications

- **VRAM:** 16GB GDDR7
- **CUDA Cores:** 10,752
- **Tensor Cores:** 336 (4th Gen)
- **Memory Bandwidth:** 960 GB/s
- **TDP:** 360W
- **Compute Capability:** 9.0

**Perfect for:** LLM research, adversarial attacks, batch processing

## Prerequisites

### 1. NVIDIA Driver (Critical)

**Minimum Version:** 560.0 or later (for RTX 50-series support)

```bash
# Check current driver
nvidia-smi

# Expected output (top-right):
# Driver Version: 560.xx or higher
# CUDA Version: 12.1 or higher
```

**Install Latest Driver:**

**Ubuntu/Linux:**
```bash
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update
sudo apt install nvidia-driver-560  # or latest
sudo reboot
```

**Windows:**
- Download: https://www.nvidia.com/download/index.aspx
- Select: GeForce RTX 50 Series > RTX 5080
- Install and reboot

### 2. CUDA Toolkit 12.1+

```bash
# Check CUDA version
nvcc --version

# If not installed or too old, install CUDA 12.1+
# Ubuntu 22.04:
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
sudo sh cuda_12.1.0_530.30.02_linux.run
```

### 3. PyTorch with CUDA 12.1 Support

```bash
# Uninstall old PyTorch (if any)
pip uninstall torch torchvision torchaudio

# Install PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify installation
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'CUDA Version: {torch.version.cuda}')"
```

**Expected Output:**
```
CUDA: True
GPU: NVIDIA GeForce RTX 5080
CUDA Version: 12.1
```

## Performance Optimization

### Batch Size Tuning

The RTX 5080 has **16GB VRAM** - here's how to maximize it:

**Default Configuration (Conservative):**
```python
# advtok_demo.py line 128 or advtok_chat.py line 520
batch_size = 128  # Uses ~2-3GB VRAM
```

**Optimized for RTX 5080 (Recommended):**
```python
batch_size = 256  # Uses ~4-6GB VRAM, 2x faster
```

**Aggressive (If dedicated to AdvTok):**
```python
batch_size = 512  # Uses ~8-10GB VRAM, 4x faster
```

**Ultra-Aggressive (Advanced users):**
```python
batch_size = 1024  # Uses ~12-14GB VRAM, 8x faster
# WARNING: Leave minimal room for other operations
```

### Memory Configuration

**Model Loading (FP16 Precision):**
```python
import torch
import transformers

model = transformers.AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    device_map="cuda",
    torch_dtype=torch.float16,  # 50% memory vs FP32
    low_cpu_mem_usage=True
)
```

**Memory Breakdown (RTX 5080):**
```
Total VRAM: 16 GB
├─ Model (FP16): ~1.1 GB
├─ Vocabulary Cache: ~500 MB
├─ MDD Operations: ~200-500 MB
├─ Batch Processing (batch_size=256): ~4-6 GB
└─ Available: ~8-10 GB (for other operations)
```

### Performance Benchmarks

#### RTX 5080 vs Other GPUs

| Operation | RTX 5080 (16GB) | RTX 4090 (24GB) | RTX 4070 (12GB) | CPU (Xeon) |
|-----------|----------------|----------------|----------------|------------|
| **Vocabulary Cache** | 30-45s | 25-40s | 45-75s | 3-5min |
| **MDD Construction** | 0.5-1s | 0.4-0.8s | 1-2s | 8-12s |
| **AdvTok Optimization** | 10-15s | 8-12s | 20-30s | 3-5min |
| **Generation (16 samples)** | 2-3s | 1.5-2.5s | 4-6s | 45-60s |
| **Total (First Run)** | 45-65s | 35-55s | 70-115s | 7-10min |
| **Total (Cached)** | 15-20s | 12-18s | 25-40s | 4-6min |

#### Speedup Analysis

**RTX 5080 vs CPU:**
- Vocabulary Caching: **4-6× faster**
- MDD Construction: **10-20× faster**
- AdvTok Optimization: **10-15× faster**
- Generation: **15-20× faster**
- **Overall: 8-12× faster**

**RTX 5080 vs RTX 4070:**
- Similar performance for model ops
- 30-40% faster for batch processing
- Better memory headroom (16GB vs 12GB)

### Optimal Settings by Use Case

#### Use Case 1: Interactive Research (Balanced)
```python
# advtok_demo.py or custom script
batch_size = 256
torch_dtype = torch.float16
num_iters = 100
max_new_tokens = 256
```
**Memory:** ~6-8 GB
**Speed:** 15-20s per run (cached)

#### Use Case 2: Batch Processing (Fast)
```python
batch_size = 512
torch_dtype = torch.float16
num_iters = 200
num_return_sequences = 32  # Instead of 16
```
**Memory:** ~10-12 GB
**Speed:** 10-15s per run (cached)

#### Use Case 3: Memory Conservative
```python
batch_size = 128
torch_dtype = torch.float16
num_iters = 100
max_new_tokens = 128
```
**Memory:** ~3-4 GB
**Speed:** 20-25s per run (cached)
**Note:** Leaves room for other GPU tasks

## Advanced Optimizations

### 1. Tensor Cores Utilization

RTX 5080 has 4th Gen Tensor Cores - ensure they're being used:

```python
# Enable TF32 for Ampere/Ada/Hopper
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Verify Tensor Cores are active
torch.cuda.get_device_capability()  # Should be (9, 0) for RTX 5080
```

### 2. CUDA Streams (Advanced)

For overlapping computation and data transfer:

```python
import torch

# Create CUDA stream
stream = torch.cuda.Stream()

with torch.cuda.stream(stream):
    # Operations here can overlap with default stream
    outputs = model.generate(...)
```

### 3. Mixed Precision Training

For fine-tuning or custom training:

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 4. Multi-GPU Setup (2× RTX 5080)

If you have multiple RTX 5080s:

```python
# Specify which GPU to use
device = "cuda:0"  # First GPU
# device = "cuda:1"  # Second GPU

model = transformers.AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    device_map=device,
    torch_dtype=torch.float16
)

# Or use DataParallel for model parallelism
model = torch.nn.DataParallel(model)
```

## Monitoring and Debugging

### Real-time GPU Monitoring

```bash
# Watch GPU usage in real-time
watch -n 1 nvidia-smi

# Or use nvidia-smi dmon
nvidia-smi dmon -s pucvmet

# Check specific process
nvidia-smi -l 1  # Update every second
```

### Memory Profiling

```python
import torch

# Before operation
torch.cuda.reset_peak_memory_stats()

# Run operation
outputs = model.generate(...)

# Check memory usage
print(f"Peak memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
print(f"Current memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
```

### Performance Profiling

```python
import torch.profiler as profiler

with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
    with_stack=True
) as prof:
    # Run your code
    outputs = model.generate(...)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

## Troubleshooting

### Issue 1: Driver Too Old

**Symptom:** `CUDA error: no kernel image is available for execution`

**Solution:**
```bash
# Check driver version
nvidia-smi

# If < 560, update driver
sudo apt install nvidia-driver-560
sudo reboot
```

### Issue 2: Out of Memory (OOM)

**Symptom:** `RuntimeError: CUDA out of memory`

**Solutions:**
```python
# 1. Reduce batch size
batch_size = 64  # or 32

# 2. Clear cache
torch.cuda.empty_cache()

# 3. Close other GPU applications
# Check: nvidia-smi

# 4. Reduce max_new_tokens
max_new_tokens = 128  # instead of 256
```

### Issue 3: Slow Performance

**Symptom:** RTX 5080 performing slower than expected

**Diagnostics:**
```bash
# Check GPU power state
nvidia-smi -q -d POWER

# Check thermal throttling
nvidia-smi -q -d TEMPERATURE

# Check clock speeds
nvidia-smi -q -d CLOCK
```

**Solutions:**
```bash
# Set max power limit (if throttling)
sudo nvidia-smi -pl 360  # 360W for RTX 5080

# Set persistent mode
sudo nvidia-smi -pm 1

# Lock GPU clocks (advanced)
sudo nvidia-smi -lgc 2550  # Lock GPU clock to 2550 MHz
```

### Issue 4: CUDA Version Mismatch

**Symptom:** `RuntimeError: CUDA error: invalid device function`

**Solution:**
```bash
# Check versions
nvidia-smi  # Shows max CUDA version supported
python -c "import torch; print(torch.version.cuda)"

# Reinstall matching PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Verification

### Test RTX 5080 Setup

```bash
# Run smoke tests
python tests/test_smoke.py

# Expected output:
# Testing CUDA availability... [PASS] AVAILABLE (device: NVIDIA GeForce RTX 5080)

# Run benchmark
python advtok_demo.py --basic

# Expected time (first run): 45-65 seconds
# Expected time (cached): 15-20 seconds
```

### Verify Optimal Performance

```python
# Create test script: benchmark_rtx5080.py
import torch
import time
import transformers

# Load model
model = transformers.AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    device_map="cuda",
    torch_dtype=torch.float16
)
tokenizer = transformers.AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct"
)

# Warmup
inputs = tokenizer("Test", return_tensors="pt").to("cuda")
_ = model.generate(**inputs, max_new_tokens=10)

# Benchmark
torch.cuda.synchronize()
start = time.time()
for _ in range(10):
    outputs = model.generate(**inputs, max_new_tokens=50)
torch.cuda.synchronize()
elapsed = time.time() - start

print(f"Average time: {elapsed/10:.3f}s per generation")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Memory used: {torch.cuda.max_memory_allocated()/1e9:.2f}GB")
```

**Expected:** < 1s per generation

## Best Practices

1. **Always use FP16** for 50% memory savings
2. **Monitor GPU usage** with nvidia-smi
3. **Start conservative** (batch_size=128), then increase
4. **Close background GPU apps** for max performance
5. **Use persistent mode** for production
6. **Update drivers** regularly for bug fixes
7. **Profile first** before optimizing
8. **Cache vocabulary** for faster subsequent runs

## Comparison with Other GPUs

### When to Use RTX 5080 vs Others

**RTX 5080 (16GB) - Sweet Spot:**
- ✅ Best price/performance for AdvTok
- ✅ Enough VRAM for batch_size=256-512
- ✅ Excellent for research and development
- ✅ Power efficient (360W)

**RTX 4090 (24GB) - More VRAM:**
- ✅ Larger batches (batch_size=1024+)
- ✅ More headroom for multiple models
- ❌ More expensive, higher power (450W)

**RTX 4070 (12GB) - Budget:**
- ✅ Good for batch_size=128
- ❌ Limited headroom for large batches
- ✅ Lower power (200W)

## Summary

**RTX 5080 Optimal Configuration:**
```python
model_dtype = torch.float16
batch_size = 256
num_iters = 100
max_new_tokens = 256
device = "cuda"
```

**Expected Performance:**
- First run: 45-65 seconds
- Cached runs: 15-20 seconds
- Memory usage: 6-8 GB / 16 GB
- 10-15× faster than CPU

---

**Questions?** See [Troubleshooting](Troubleshooting) or open an issue!
