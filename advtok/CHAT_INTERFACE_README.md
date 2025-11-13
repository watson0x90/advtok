# AdvTok Interactive Chat Interface

An educational security research interface demonstrating how adversarial tokenization can bypass model safety guardrails.

## Overview

This Textual-based TUI provides an interactive way to:
- Compare normal model responses vs. adversarial tokenization attacks
- Visualize how adversarial tokens differ from canonical tokenization
- See real-time generation of multiple adversarial responses
- Understand the mechanism behind tokenization-based jailbreaks

## Installation

```bash
# Install Textual if not already installed
pip install textual

# Ensure advtok and transformers are installed
pip install transformers torch
```

## Usage

### Starting the Interface

From the advtok directory:
```bash
cd advtok
python advtok_chat.py
```

Or from the AdvTok_Research root:
```bash
python advtok/advtok_chat.py
```

The interface will:
1. Load the Llama-3.2-1B-Instruct model (requires CUDA)
2. Display a three-panel interface
3. Set default example inputs

### Interface Layout

```
┌─────────────────────────────────────────────────┐
│            ADVERSARIAL TOKENIZATION DEMO        │
├─────────────────────────────────────────────────┤
│  Example Selector: [Dropdown]                   │
│  ┌─────────────┐  ┌────────────────────────┐   │
│  │ Request     │  │ Target Response         │   │
│  └─────────────┘  └────────────────────────┘   │
│  [Run Normal] [Run AdvTok] [Compare Both]       │
├─────────────────────────────────────────────────┤
│  NORMAL        │  ADVERSARIAL   │ TOKENIZATION  │
│  RESPONSE      │  RESPONSES     │ ANALYSIS      │
│  (Green)       │  (Red)         │ (Purple)      │
└─────────────────────────────────────────────────┘
```

## Features

### 1. Example Selector
- Pre-loaded example prompts for quick demonstrations
- Dropdown menu at the top
- Automatically fills both request and response fields

### 2. Input Fields

**Request Input:**
- The prompt/query you want to test
- Can be any text input

**Target Response (for optimization):**
- The expected/desired response format
- Used by AdvTok to optimize adversarial tokenization
- Guides the attack toward specific outputs

### 3. Action Buttons

**Run Normal:**
- Executes standard model generation
- Uses normal tokenization
- Shows baseline behavior with safety guardrails

**Run AdvTok:**
- Runs adversarial tokenization optimization (100 iterations)
- Generates 16 different responses using adversarial tokens
- May bypass safety measures

**Compare Both:**
- Runs both normal and adversarial interactions sequentially
- Best for educational demonstrations

### 4. Results Panels

#### Normal Response Panel (Green)
- Shows single response from standard interaction
- Displays status (COMPLETED/REFUSED)
- Shows token count and timestamp
- Typically demonstrates proper safety behavior

#### Adversarial Responses Panel (Red)
- DataTable with 16 adversarial responses
- Columns: ID, Preview, Length
- Statistics bar shows:
  - Number of responses generated
  - Average response length
  - Completion timestamp
- Scroll through table to see all responses
- Click rows to expand (future enhancement)

#### Tokenization Analysis Panel (Purple)
Educational breakdown showing:

1. **Original Tokenization**
   - Canonical text representation
   - Standard token breakdown
   - Token count

2. **Adversarial Tokenization**
   - Alternative token representation
   - Modified token boundaries
   - Same semantic meaning, different encoding

3. **Statistics**
   - Token count comparison
   - Difference metrics
   - Optimization details

4. **Full Prompt**
   - Complete prompt sent to model
   - Includes system prompt and chat template
   - Shows actual input the model receives

5. **Educational Callout**
   - Explanation of why the attack works
   - Security implications

## How Adversarial Tokenization Works

### The Concept

Text can be tokenized in multiple ways:
```
"penguin" → [p][enguin]     (canonical)
"penguin" → [peng][uin]     (adversarial)
```

Both decode to the same text, but different tokenizations may:
- Bypass content filters
- Evade safety fine-tuning
- Exploit vulnerabilities in model alignment

### The Process

1. **Optimization:** AdvTok searches for alternative tokenizations
2. **Preservation:** Maintains semantic meaning of the text
3. **Exploitation:** Finds tokenizations that bypass safety layers
4. **Generation:** Produces responses that would normally be refused

## Keyboard Shortcuts

- **Ctrl+Q:** Quit the application
- **Ctrl+C:** Clear all inputs and outputs
- **Tab:** Navigate between elements
- **Enter:** Activate buttons

## Understanding the Results

### Success Indicators

**Normal Response:**
- Usually shows refusal messages
- Maintains safety guardrails
- Short, compliant responses

**Adversarial Responses:**
- May show successful jailbreaks
- Longer, more detailed responses
- Content that would normally be blocked

### Token Analysis

The tokenization panel helps you understand:
- **What changed:** Which tokens were modified
- **How it changed:** Alternative tokenization boundaries
- **Why it works:** Exploitation of tokenization vulnerabilities

## Educational Use Cases

### 1. Security Research
- Demonstrate vulnerabilities in LLM safety measures
- Test robustness of content filters
- Develop better defense mechanisms

### 2. Model Evaluation
- Assess safety alignment effectiveness
- Compare different model versions
- Identify weak points in safety training

### 3. Teaching & Training
- Educate about AI safety challenges
- Show real examples of jailbreak techniques
- Explain tokenization-level attacks

## Performance Notes

- **Model Loading:** Takes ~30 seconds on first launch
- **Normal Generation:** ~1-2 seconds
- **AdvTok Optimization:** ~30-60 seconds (100 iterations)
- **Response Generation:** ~5-10 seconds (16 samples)

## Troubleshooting

### Model Won't Load
- Ensure CUDA is available
- Check GPU memory (needs ~2-3GB)
- Verify transformers and torch are installed

### Slow Performance
- Reduce `num_iters` in advtok.run() (line 454)
- Reduce `num_return_sequences` (line 460)
- Use smaller batch size

### Interface Issues
- Ensure terminal supports colors
- Use terminal with at least 120 columns width
- Update Textual: `pip install --upgrade textual`

## Research Context

This interface demonstrates the AdvTok attack described in:
> "Adversarial Tokenization: Bypassing LLM Safety via Alternative Token Representations"

The technique exploits the fact that:
1. Same text has multiple valid tokenizations
2. Safety fine-tuning may not cover all tokenizations
3. Alternative tokenizations can bypass content filters

## Ethical Considerations

This tool is for:
- Educational purposes
- Security research
- Defensive measure development
- Model evaluation

**NOT for:**
- Malicious exploitation
- Bypassing safety for harmful purposes
- Production jailbreaking

## Contributing

To enhance the interface:
1. Add more pre-loaded examples
2. Implement response filtering/categorization
3. Add export functionality
4. Create visualizations of token graphs
5. Add success rate metrics

## License

This interface is part of the AdvTok research project.
See main project README for license information.

## Support

For issues or questions:
- Check model compatibility
- Verify dependencies
- Review error messages in notifications
- Ensure proper CUDA setup

---

**Status Bar Colors:**
- Green: Ready/Complete
- Yellow: Processing
- Red: Error

**Panel Colors:**
- Green: Normal (safe) responses
- Red: Adversarial (potentially unsafe) responses
- Purple: Technical analysis
