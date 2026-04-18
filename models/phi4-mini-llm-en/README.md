---
license: apache-2.0
base_model: microsoft/Phi-4-mini-reasoning
library_name: peft
tags:
  - lora
  - domain-adaptation
  - business
  - finance
  - erp
  - causal-lm
  - phi-4
language:
  - en
pipeline_tag: text-generation
model-index:
  - name: kniv-phi4-mini-llm-en
    results: []
---

# kniv-phi4-mini-llm-en

Business-domain language model produced by LoRA fine-tuning of
[microsoft/Phi-4-mini-reasoning](https://huggingface.co/microsoft/Phi-4-mini-reasoning)
on finance, ERP, supply-chain, accounting, and marketing text.
The adapters are roughly 50 MB and can be merged into the base model at
inference time.

## Model Details

| Field | Value |
|-------|-------|
| **Developer** | [dragonscale-ai](https://huggingface.co/dragonscale-ai) |
| **Base model** | `microsoft/Phi-4-mini-reasoning` (3.8B parameters) |
| **Base model license** | MIT |
| **Adapter license** | Apache-2.0 |
| **Method** | LoRA continued pretraining (causal language modeling) |
| **Language** | English |
| **Repository** | [rustic-ai/kniv-nlp-models](https://github.com/rustic-ai/kniv-nlp-models) |

### Intended Use

Domain-adapted business reasoning for the
[uniko](https://github.com/rustic-ai/uniko) cognitive memory system.
The model is designed to improve perplexity and generation quality on
business, finance, ERP, supply-chain, accounting, and marketing text
compared to the general-purpose base model.

## Training Data

Full documents (not sentence-split) from seven open sources, preserving
discourse structure. Documents are chunked into **4096-token** sequences with
**128-token overlap**.

| Source | License | Content |
|--------|---------|---------|
| SEC EDGAR 10-K | Public domain | Financial filings, MD&A, risk factors |
| Enron Emails | Public domain | Business communication |
| OpenStax Textbooks | CC BY-4.0 | Management, accounting, entrepreneurship |
| Odoo Documentation | CC BY-SA 3.0 | ERP module documentation |
| Wikipedia Business Articles | CC BY-SA 3.0 | Business encyclopedia articles |
| CUAD Contracts | CC BY-4.0 | Commercial contracts |
| OpenAlex Abstracts | ODC-BY | Academic business and finance papers |

## LoRA Configuration

| Parameter | Value |
|-----------|-------|
| Rank (`r`) | 16 |
| Alpha (`lora_alpha`) | 32 |
| Dropout (`lora_dropout`) | 0.05 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Task type | `CAUSAL_LM` |
| Adapter size | ~50 MB |

### Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Epochs | 3 |
| Per-device batch size | 4 |
| Gradient accumulation steps | 8 (effective batch size 32) |
| Learning rate | 2e-4 |
| LR scheduler | Cosine |
| Warmup ratio | 0.05 |
| Weight decay | 0.01 |
| Max gradient norm | 1.0 |
| Precision | bfloat16 |
| Max sequence length | 4096 |
| Dev split | 5% |

## Usage

### Loading the Adapters (Python)

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-4-mini-reasoning",
    torch_dtype="auto",
    device_map="auto",
)
model = PeftModel.from_pretrained(base, "dragonscale-ai/kniv-phi4-mini-llm-en")
model = model.merge_and_unload()  # merge LoRA weights into base model

tokenizer = AutoTokenizer.from_pretrained("dragonscale-ai/kniv-phi4-mini-llm-en")

inputs = tokenizer("The company reported quarterly revenue of", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Training from Source

```bash
# 1. Prepare data (tokenize + chunk documents)
python models/phi4-mini-llm-en/prepare_data.py

# 2. Train (LoRA fine-tuning)
python models/phi4-mini-llm-en/train.py

# 3. Resume from checkpoint if interrupted
python models/phi4-mini-llm-en/train.py --resume outputs/phi4-mini-llm-en/checkpoint-1000
```

## Alternate Base Models

The training pipeline supports swapping the base model by changing `model.base`
in `config.yaml`. The same LoRA configuration and corpus are used; only the
base weights differ.

| Model | Parameters | License | Strength |
|-------|-----------|---------|----------|
| `microsoft/Phi-4-mini-reasoning` | 3.8B | MIT | Best reasoning under 4B (default) |
| `Qwen/Qwen3-4B` | 4B | Apache 2.0 | Thinking-mode toggle, 262K context |
| `google/gemma-4-E4B` | 4B effective | Apache 2.0 | MoE architecture, multimodal |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | 7B | MIT | Strongest small reasoner |

## Limitations

- **Domain scope.** The model is adapted for business, finance, ERP, and
  related domains. Performance on general-purpose or out-of-domain tasks may
  be comparable to or slightly worse than the unmodified base model.
- **Not instruction-tuned.** This is a continued-pretraining adapter, not an
  instruction-following or chat model. Wrap prompts in the base model's chat
  template if conversational output is needed.
- **English only.** Training data is exclusively English. Multilingual
  capability is limited to whatever the base model retains.
- **No safety fine-tuning.** The adapter does not add safety guardrails beyond
  those already present in the base model.
- **Hallucination risk.** Like all language models, this model can generate
  plausible-sounding but factually incorrect text. Do not rely on its output
  for financial, legal, or regulatory decisions without independent
  verification.
