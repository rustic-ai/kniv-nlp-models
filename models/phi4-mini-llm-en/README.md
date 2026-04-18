# uniko-phi4-mini-llm-en

Business domain LLM — continued pretraining of Phi-4-mini-reasoning on finance, ERP, supply chain, and management text.

## Model Details

- **Base:** `microsoft/Phi-4-mini-reasoning` (3.8B, MIT license)
- **Method:** LoRA fine-tuning (rank 16, ~50MB adapters)
- **Domain:** Business, finance, ERP, supply chain, accounting, marketing
- **Intended use:** Domain-adapted reasoning for the [uniko](https://github.com/rustic-ai/uniko) cognitive memory system

## Alternate Base Models

Change `model.base` in `config.yaml` to switch:

| Model | Params | License | Strength |
|-------|--------|---------|----------|
| `microsoft/Phi-4-mini-reasoning` | 3.8B | MIT | Best reasoning under 4B (default) |
| `Qwen/Qwen3-4B` | 4B | Apache 2.0 | Thinking mode toggle, 262K context |
| `google/gemma-4-E4B` | 4B eff. | Apache 2.0 | MoE, multimodal |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | 7B | MIT | Strongest small reasoner |

## Training Data

Full documents (not sentence-split) from 7 open sources:

| Source | License | Content |
|--------|---------|---------|
| SEC EDGAR 10-K | Public domain | Financial filings, MD&A, risk factors |
| Enron Emails | Public domain | Business communication |
| OpenStax Textbooks | CC BY-4.0 | Management, accounting, entrepreneurship |
| Odoo Documentation | CC BY-SA 3.0 | ERP module docs |
| Wikipedia Business | CC BY-SA 3.0 | Business encyclopedia articles |
| CUAD Contracts | CC BY-4.0 | Commercial contracts |
| OpenAlex Abstracts | ODC-BY | Academic business/finance papers |

## Usage

```bash
# 1. Prepare data (tokenize + chunk documents)
python models/phi4-mini-llm-en/prepare_data.py

# 2. Train (LoRA fine-tuning)
python models/phi4-mini-llm-en/train.py

# 3. Resume from checkpoint if interrupted
python models/phi4-mini-llm-en/train.py --resume outputs/phi4-mini-llm-en/checkpoint-1000
```

## Loading the fine-tuned model

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("microsoft/Phi-4-mini-reasoning")
model = PeftModel.from_pretrained(base, "outputs/phi4-mini-llm-en/final")
model = model.merge_and_unload()  # merge LoRA into base weights

tokenizer = AutoTokenizer.from_pretrained("outputs/phi4-mini-llm-en/final")
```
