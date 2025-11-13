# Installation Guide

Complete instructions for installing AdvTok Research on your system.

## System Requirements

### Hardware Requirements
- **CPU:** Any modern multi-core processor
- **RAM:** Minimum 8GB (16GB+ recommended)
- **GPU (Optional but Recommended):**
  - NVIDIA GPU with CUDA support
  - RTX 5080 (16GB) - Optimal
  - RTX 4090 (24GB) - Excellent
  - RTX 4080/4070 (12-16GB) - Good
  - Minimum 8GB VRAM for comfortable use
- **Storage:** ~10GB free space (for model cache)

### Software Requirements
- **Operating System:**
  - Linux (Ubuntu 20.04+, tested)
  - Windows 10/11 (tested)
  - macOS (CPU only, untested)
- **Python:** 3.8 or later (3.10 recommended)
- **CUDA:** 12.1+ (for GPU acceleration)
- **NVIDIA Driver:** 560+ for RTX 50-series, 525+ for older GPUs

## Installation Methods

### Method 1: Quick Install (Recommended)

```bash
# Clone the repository
git clone -b research https://github.com/watson0x90/advtok.git
cd advtok

# Install dependencies
pip install -r advtok/requirements.txt

# Verify installation
cd advtok
python tests/test_smoke.py
```

### Method 2: With Virtual Environment (Recommended for Production)

```bash
# Clone the repository
git clone -b research https://github.com/watson0x90/advtok.git
cd advtok

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r advtok/requirements.txt

# Verify installation
cd advtok
python tests/test_smoke.py
```

### Method 3: Development Install

```bash
# Clone the repository
git clone -b research https://github.com/watson0x90/advtok.git
cd advtok

# Install in editable mode
cd advtok
pip install -e .

# Install development dependencies
pip install pytest pytest-cov black flake8

# Run tests
python tests/test_smoke.py
python tests/test_advtok_stability.py
```

## GPU Setup (NVIDIA CUDA)

### Step 1: Install NVIDIA Driver

**For RTX 5080 (50-series):**
```bash
# Ubuntu/Linux
sudo apt update
sudo apt install nvidia-driver-560  # or later

# Verify
nvidia-smi  # Should show your GPU
```

**Windows:**
- Download from: https://www.nvidia.com/download/index.aspx
- Select: GeForce RTX 50 Series
- Install and reboot

### Step 2: Install CUDA Toolkit

```bash
# Ubuntu 22.04
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda-repo-ubuntu2204-12-1-local_12.1.0-530.30.02-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2204-12-1-local_12.1.0-530.30.02-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2204-12-1-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda

# Verify
nvcc --version
```

**Windows:**
- Download from: https://developer.nvidia.com/cuda-downloads
- Run installer
- Add to PATH if not done automatically

### Step 3: Install PyTorch with CUDA Support

```bash
# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify PyTorch CUDA support
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}'); print(f'Device: {torch.cuda.get_device_name(0)}')"
```

**Expected Output:**
```
CUDA Available: True
CUDA Version: 12.1
Device: NVIDIA GeForce RTX 5080
```

## Dependencies

The `requirements.txt` includes:

```
torch>=2.0.0
transformers>=4.30.0
textual>=0.40.0
rich>=13.0.0
numpy>=1.24.0
tqdm>=4.65.0
```

## Verification

### Run Smoke Tests (Quick - 1 second)

```bash
cd advtok
python tests/test_smoke.py
```

**Expected Output:**
```
========================================
AdvTok Smoke Tests
========================================

Testing basic imports... [PASS]
Testing advtok package structure... [PASS]
Testing tokenizer import... [PASS]
Testing model import... [PASS]
Testing async functionality... [PASS]
Testing signal handling... [PASS]
Testing timeout mechanisms... [PASS]
Testing multiprocessing availability... [PASS]
Testing vocabulary cache structure... [PASS]
Testing MDD basic structure... [PASS]
Testing CUDA availability... [PASS] AVAILABLE (device: NVIDIA GeForce RTX 5080)

Results: 11/11 tests passed (100.0%)
[PASS] All smoke tests passed! The system is ready.
```

### Run Comprehensive Tests (15 seconds)

```bash
python tests/test_advtok_stability.py
```

**Expected:** 25+ tests passing

## Troubleshooting

### Issue: Import Error

**Error:** `ModuleNotFoundError: No module named 'advtok'`

**Solution:**
```bash
# Make sure you're in the advtok directory
cd advtok

# Or install as package
cd ..
pip install -e advtok
```

### Issue: CUDA Not Available

**Error:** `torch.cuda.is_available()` returns `False`

**Solution:**
```bash
# Check NVIDIA driver
nvidia-smi

# Reinstall PyTorch with CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify
python -c "import torch; print(torch.cuda.is_available())"
```

### Issue: Version Conflicts

**Error:** Dependency version conflicts

**Solution:**
```bash
# Create fresh virtual environment
python -m venv fresh_venv
source fresh_venv/bin/activate  # or venv\Scripts\activate on Windows

# Install with no-cache
pip install --no-cache-dir -r requirements.txt
```

### Issue: Out of Memory

**Error:** `RuntimeError: CUDA out of memory`

**Solution:**
```bash
# Clear CUDA cache
python -c "import torch; torch.cuda.empty_cache()"

# Use smaller batch size (edit advtok_demo.py line 128)
batch_size = 64  # or 32
```

## Next Steps

After successful installation:

1. **Quick Start:** See [Quick Start Guide](Quick-Start)
2. **GPU Optimization:** See [RTX 5080 Setup](RTX-5080-Setup)
3. **Run Demo:** See [Using Demo Script](Using-Demo-Script)
4. **Learn Concepts:** See [How It Works](How-It-Works)

## Additional Resources

- **PyTorch Installation:** https://pytorch.org/get-started/locally/
- **CUDA Toolkit:** https://developer.nvidia.com/cuda-downloads
- **Transformers Docs:** https://huggingface.co/docs/transformers/
- **GPU Troubleshooting:** [Troubleshooting Guide](Troubleshooting)

---

**Need Help?** Open an issue: https://github.com/watson0x90/advtok/issues
