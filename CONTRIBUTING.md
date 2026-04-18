# Contributing to kniv-nlp-models

This project trains NLP and LLM models for the uniko cognitive memory system. Contributions are welcome across corpus domains, model architectures, and tooling.

## Development Setup

```bash
git clone git@github.com:rustic-ai/kniv-nlp-models.git
cd kniv-nlp-models
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev,corpus,llm]"
```

Install the eval extras if you need spaCy-based evaluation:

```bash
uv pip install -e ".[eval]"
```

## Adding a New Corpus Domain

Each domain lives under `corpus/domains/<domain_name>/` and must contain:

| File | Purpose |
|------|---------|
| `collect.py` | Fetch or generate raw text from sources |
| `preprocess.py` | Clean, sentence-split, and format into annotation-ready JSONL |
| `config.yaml` | Domain-specific settings (sources, filters, target counts) |
| `README.md` | Data provenance, licensing, and processing notes |

After creating the directory, register the domain in `corpus/domains/domain_config.yaml`:

```yaml
your_domain:
  status: pending
  target_sentences: 15000
  priority: medium
  description: "Brief description of the domain and its relevance"
```

Status values: `pending` | `collected` | `annotated` | `validated` | `published`.

## Adding a New Model

Each model lives under `models/<model_name>/` and must contain:

| File | Purpose |
|------|---------|
| `config.yaml` | Hyperparameters, base model, task heads, label maps |
| `model.py` | Model architecture and task-specific heads |
| `train.py` | Training loop with evaluation and checkpointing |
| `prepare_data.py` | Convert annotated corpus into model-ready datasets |
| `README.md` | Architecture decisions, benchmark results, usage notes |

Follow existing models (e.g., `models/deberta-v3-nlp-en/`) as a reference for structure and conventions.

## Running Tests

```bash
uv run pytest -n auto
```

Run a specific test file or directory:

```bash
uv run pytest tests/test_pipeline.py -n auto
```

## Code Style

This project uses **ruff** for linting and formatting:

```bash
uv run ruff check .
uv run ruff format .
```

Guidelines:
- Add type hints to all function signatures
- Write docstrings for public functions and classes
- Keep functions focused and under ~50 lines where practical
- Use `pathlib.Path` over `os.path`

## Commit Conventions

- Use imperative mood: "Add domain" not "Added domain" or "Adds domain"
- Be descriptive: "Add business corpus domain with SEC filing collector" not "Add stuff"
- Keep the subject line under 72 characters
- Reference issues when applicable: "Fix sentence splitting for nested quotes (#42)"

## Pull Requests

- One feature or fix per PR
- Include a test plan in the PR description (what you tested, how to verify)
- Ensure `uv run pytest -n auto` passes before opening the PR
- Ensure `uv run ruff check .` reports no errors
- Keep PRs small and reviewable; split large changes into a series of PRs
