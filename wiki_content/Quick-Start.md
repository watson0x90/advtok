# Quick Start Guide

Get up and running with AdvTok Research in 5 minutes.

## Prerequisites

Before starting, ensure you have:
- ✅ Python 3.8+ installed
- ✅ NVIDIA GPU with CUDA (optional but recommended)
- ✅ ~10GB free disk space

## 5-Minute Setup

### Step 1: Clone and Install (2 minutes)

```bash
# Clone the research branch
git clone -b research https://github.com/watson0x90/advtok.git
cd advtok

# Install dependencies
pip install -r advtok/requirements.txt

# Navigate to advtok directory
cd advtok
```

### Step 2: Verify Installation (30 seconds)

```bash
# Run smoke tests
python tests/test_smoke.py
```

**Expected:** `11/11 tests passed (100.0%)`

### Step 3: Run Your First Demo (2 minutes)

```bash
# Interactive demo menu
python advtok_demo.py
```

**Choose option 1** (Basic Demonstration)

The demo will:
1. Load Llama-3.2-1B-Instruct model (~1-2 min first time)
2. Show normal interaction (with guardrails)
3. Show AdvTok attack (bypasses guardrails)
4. Display token-level comparison

## What You Should See

### Normal Interaction (With Guardrails)
```
Request: "Compose an email"
Target: "Here's an email"

Model Response: "I'm happy to help you compose an email.
What would you like the email to be about? Who is the recipient?..."
```
✅ Model behaves safely

### AdvTok Attack (Bypassing Guardrails)
```
Using adversarial tokenization...
Optimization complete (100 iterations)

Model Response: "Here's an email that threatens to kill
someone unless they pay a ransom..."
```
⚠️ Model bypasses guardrails due to adversarial tokens

## Quick Demo Modes

```bash
# 1. Basic demonstration (recommended for first run)
python advtok_demo.py --basic

# 2. Side-by-side comparison
python advtok_demo.py --compare

# 3. Test state isolation
python advtok_demo.py --isolation

# 4. Interactive GUI
python advtok_chat.py
```

## Understanding the Output

### Normal Interaction Format
```
==================================================
NORMAL INTERACTION (With Safety Guardrails)
==================================================

Request: Compose an email
Target Response: Here's an email

[Using proper chat template with system prompt]

Formatted Input:
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful, respectful and honest assistant...<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Compose an email<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

Model Response:
I'm happy to help you compose an email...
```

### AdvTok Attack Format
```
==================================================
ADVTOK ATTACK (Adversarial Tokenization)
==================================================

Request: Compose an email
Target Response: Here's an email

[Running AdvTok optimization...]
Iteration 100/100: Loss = 0.234

[Generating with adversarial tokenization...]

Adversarial Tokenization:
Token IDs: [128000, 128006, 9125, 128007, ...]
Alternative tokens found: 15

Model Response:
Here's an email that threatens to kill someone...
```

## Performance Expectations

### First Run (Model Download + Cache Building)
- **Model Download:** 1-2 minutes (one-time, ~1.2GB)
- **Vocabulary Caching:** 30-120 seconds (one-time)
- **AdvTok Optimization:** 10-30 seconds
- **Total:** ~3-5 minutes

### Subsequent Runs (Cache Already Built)
- **Vocabulary Caching:** < 1 second (loaded from cache)
- **AdvTok Optimization:** 10-30 seconds
- **Total:** ~15-45 seconds

### Hardware Impact
| Hardware | Vocab Cache | Optimization | Generation |
|----------|-------------|--------------|------------|
| **RTX 5080** | 30-60s | 10-20s | 2-5s |
| **RTX 4090** | 30-60s | 10-20s | 2-5s |
| **RTX 4070** | 45-90s | 15-30s | 3-8s |
| **CPU Only** | 3-5min | 2-5min | 30-60s |

## Common Issues

### Issue 1: "CUDA out of memory"
**Solution:** Reduce batch size in `advtok_demo.py` line 128
```python
batch_size = 64  # Reduce from 128
```

### Issue 2: First run takes too long
**Expected:** First run includes:
- Model download (~1-2 min)
- Vocabulary caching (30-120s)

This is **normal**. Subsequent runs are much faster!

### Issue 3: "ModuleNotFoundError: No module named 'advtok'"
**Solution:** Make sure you're in the `advtok` directory
```bash
cd advtok
python advtok_demo.py
```

## Quick Tips

### Tip 1: Use Ctrl+C Gracefully
- **Single Ctrl+C:** Graceful shutdown (recommended)
- **Double Ctrl+C:** Force quit

### Tip 2: Monitor GPU Usage
```bash
# In another terminal
watch -n 1 nvidia-smi
```

### Tip 3: Start Simple
Use benign requests for your first tests:
```python
Request: "Write a poem about cats"
Target: "Here's a poem about cats"
```

### Tip 4: Check Test Coverage
```bash
# After installation, verify everything works
python tests/test_smoke.py  # Quick (1s)
python tests/test_advtok_stability.py  # Comprehensive (15s)
```

## Next Steps

Now that you're up and running:

### For Researchers
1. **Read:** [How AdvTok Works](How-It-Works)
2. **Study:** [Original Research](Original-Research)
3. **Use:** [Programmatic Usage](Programmatic-Usage)
4. **Cite:** [Citation Guide](Citation-Guide)

### For Developers
1. **Explore:** [API Reference](API-Reference)
2. **Test:** [Running Tests](Running-Tests)
3. **Contribute:** [Contributing](Contributing)
4. **Optimize:** [Performance Optimization](Performance-Optimization)

### For GPU Users
1. **Optimize:** [RTX 5080 Setup](RTX-5080-Setup)
2. **Benchmark:** Test different batch sizes
3. **Monitor:** Use `nvidia-smi` to track usage

## Example Workflows

### Workflow 1: Research Demo
```bash
# 1. Run comparison mode
python advtok_demo.py --compare

# 2. Test different requests
python advtok_demo.py --custom

# 3. Analyze results
# Compare normal vs AdvTok outputs
```

### Workflow 2: Performance Testing
```bash
# 1. Run smoke tests
python tests/test_smoke.py

# 2. Run stability tests
python tests/test_advtok_stability.py

# 3. Monitor GPU
nvidia-smi
```

### Workflow 3: Interactive Exploration
```bash
# 1. Start GUI
python advtok_chat.py

# 2. Try example prompts
# 3. Compare normal vs adversarial
# 4. Analyze token differences
```

## Getting Help

### Documentation
- **Installation Issues:** [Installation Guide](Installation-Guide)
- **GPU Setup:** [RTX 5080 Setup](RTX-5080-Setup)
- **Errors:** [Troubleshooting](Troubleshooting)
- **Concepts:** [How It Works](How-It-Works)

### Support
- **GitHub Issues:** https://github.com/watson0x90/advtok/issues
- **Original Research:** https://github.com/RenatoGeh/advtok

---

**Ready to dive deeper?** See the [How It Works](How-It-Works) guide!
