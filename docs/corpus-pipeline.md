# Corpus Building Pipeline

The corpus pipeline builds a multi-domain, linguistically annotated dataset for training NLP models. It processes raw text from diverse sources into CoNLL-U formatted files with POS, NER, and dependency annotations.

## Pipeline overview

```
collect --> preprocess --> annotate --> validate --> export
(per-domain)  (per-domain)   (shared)     (shared)    (shared)
```

Each domain has its own `collect.py` and `preprocess.py` under `corpus/domains/{domain}/`. The annotate, validate, and export stages are shared across all domains and live in `corpus/pipeline/`.

## Directory structure

```
corpus/
  domains/
    conversation/       # collect.py, preprocess.py, config.yaml
    narrative/
    business/
    technical/
    news/
    encyclopedic/
    domain_config.yaml  # registry of all domains and their status
  pipeline/
    config.py           # shared paths, model names, thresholds
    annotate.py         # spaCy annotation
    validate.py         # LLM-based validation
    export.py           # merge, shuffle, split
    stats.py            # corpus statistics
  output/
    raw/{domain}/       # raw collected data
    annotated/{domain}/ # spaCy annotations (CoNLL-U + JSONL)
    validated/{domain}/ # validation results and corrections
    final/              # train.conllu, dev.conllu, test.conllu
```

## Stage 1: Collect

Each domain downloads or fetches raw text from its configured sources. Collection is idempotent -- if output files already exist, the source is skipped.

```bash
# Collect a single domain
uv run python -m corpus.domains.conversation.collect
uv run python -m corpus.domains.narrative.collect
uv run python -m corpus.domains.business.collect
uv run python -m corpus.domains.technical.collect
uv run python -m corpus.domains.news.collect
uv run python -m corpus.domains.encyclopedic.collect

# Collect a specific source within business
uv run python -m corpus.domains.business.collect --source sec_edgar
```

Raw data is written to `corpus/output/raw/{domain}/`, with each source saving to its own file or subdirectory.

## Stage 2: Preprocess

Each domain's `preprocess.py` reads raw data, splits text into sentences, cleans and filters them, deduplicates, and outputs a single `sentences.jsonl` file.

```bash
uv run python -m corpus.domains.conversation.preprocess
uv run python -m corpus.domains.narrative.preprocess
uv run python -m corpus.domains.business.preprocess
uv run python -m corpus.domains.technical.preprocess
uv run python -m corpus.domains.news.preprocess
uv run python -m corpus.domains.encyclopedic.preprocess
```

**Processing steps:**

- **Sentence splitting** -- regex-based splitting on `.!?` with abbreviation awareness (handles `Mr.`, `U.S.`, `e.g.`, decimal numbers, ellipsis).
- **Text cleaning** -- whitespace normalization, unmatched quote removal. Email bodies get additional cleaning (strip forwarded content, reply chains, signatures).
- **Quality filtering** -- minimum 5 words, maximum 100-120 words (domain-dependent), minimum 15 characters, minimum alpha ratio of 0.6-0.7. Business text uses a lower alpha threshold to accommodate numbers. All-caps lines, tables, and lines with excessive pipes or tabs are rejected.
- **Deduplication** -- case-insensitive exact match.

**Output format** (`sentences.jsonl`):

```json
{"text": "The quarterly report showed a 15% increase in revenue.", "source": "sec_edgar/Apple Inc", "domain": "business"}
```

## Stage 3: Annotate

Annotation uses spaCy's transformer-based model to produce POS tags, NER spans, and dependency parses for every sentence.

```bash
uv run python -m corpus.pipeline.annotate --domain conversation
uv run python -m corpus.pipeline.annotate --domain business --batch-size 200
```

**What it does:**

1. Loads `sentences.jsonl` from `corpus/output/raw/{domain}/`.
2. Runs all sentences through `en_core_web_trf` using `nlp.pipe()` with configurable batch size (default 100).
3. In a single pass, writes both output formats simultaneously.

**Output files** (in `corpus/output/annotated/{domain}/`):

- `annotated.conllu` -- standard CoNLL-U with NER encoded as BIO tags in the MISC column.
- `annotated.jsonl` -- structured JSON with tokens, NER spans, and metadata.

**CoNLL-U columns:** ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC (NER).

Example CoNLL-U output:

```
# sent_id = business-000042
# text = Apple reported record revenue of $124 billion.
1   Apple      Apple      PROPN  NNP  _  2  nsubj   _  NER=B-ORG
2   reported   report     VERB   VBD  _  0  ROOT    _  _
3   record     record     NOUN   NN   _  4  compound _  _
4   revenue    revenue    NOUN   NN   _  2  dobj    _  _
...
```

## Stage 4: Validate

Validation sends annotated sentences to an LLM for quality checking. The LLM reviews POS tags, NER spans, and dependency relations, flagging clear errors.

```bash
uv run python -m corpus.pipeline.validate --domain conversation
uv run python -m corpus.pipeline.validate --domain business --resume
```

**Configuration:**

- **Model:** GPT-5.4-nano (via OpenAI API, requires `OPENAI_API_KEY` env var)
- **Concurrency:** 50 simultaneous async requests
- **Chunk size:** 500 sentences per progress checkpoint
- **Resumable:** `--resume` skips successfully validated sentences and retries only those that errored

**Output files** (in `corpus/output/validated/{domain}/`):

- `validation_results.jsonl` -- per-sentence status (`ok` or `error`) with corrections or error messages.
- `corrections.json` -- aggregated map of sentence IDs to their corrections, extracted from results.

Each sentence is formatted as a structured prompt showing token IDs, forms, POS tags, heads, dependency relations, and NER spans. The LLM returns JSON with a `corrections` array specifying the token index, field (`pos`/`ner`/`dep`), old value, and new value.

## Stage 5: Export

Export merges annotations from all specified domains into a single corpus, shuffles deterministically, and splits into train/dev/test sets.

```bash
uv run python -m corpus.pipeline.export \
  --domains conversation narrative business technical news encyclopedic
```

**Processing:**

1. Loads `annotated.conllu` from each domain.
2. Merges all sentences into a single list.
3. Shuffles with a fixed seed (42) for reproducibility.
4. Splits 80/10/10 into train, dev, and test sets.

**Output files** (in `corpus/output/final/`):

- `train.conllu`
- `dev.conllu`
- `test.conllu`
- `metadata.json` -- domain list, split sizes, and total sentence count.

## Corpus statistics

Use the stats module to inspect annotated or exported data:

```bash
# Stats for a single domain
uv run python -m corpus.pipeline.stats --domain conversation

# Stats for the final exported corpus
uv run python -m corpus.pipeline.stats --final
```

Reports include sentence count, token count, average sentence length, POS distribution, dependency relation distribution, and NER span counts.

## Domains

### conversation

- **Source:** DailyDialog (13,118 dialogues, HuggingFace Datasets)
- **License:** CC BY-NC-SA 4.0
- **Target:** 30,000 sentences
- **Description:** Daily conversational dialogue across 10 topics

### narrative

- **Source:** Project Gutenberg (selected public domain novels, HTTP download)
- **License:** Public domain
- **Target:** 20,000 sentences
- **Description:** Fiction and personal stories

### business

- **Sources (7):**
  - SEC EDGAR -- 10-K filings via EDGAR API (public domain)
  - Enron email corpus -- CMU archive download (public domain)
  - OpenStax -- business textbooks via git clone, CNXML extraction (CC BY-4.0)
  - Odoo -- ERP documentation via git clone, RST extraction (CC BY-SA 3.0)
  - Wikipedia -- business category articles via API (CC BY-SA 3.0)
  - CUAD -- commercial contract clauses via HuggingFace Datasets (CC BY-4.0)
  - OpenAlex -- academic abstracts in supply chain, corporate finance, operations management, marketing, accounting (ODC-BY)
- **Target:** 200,000 sentences
- **Description:** ERP, finance, supply chain, management, marketing, and enterprise text
- **Notes:** Individual sources can be collected selectively with `--source`. Sources are independent -- failures in one do not block others.

### technical

- **Sources:**
  - Wikipedia -- computer science category articles via API (CC BY-SA 3.0)
  - Python documentation -- CPython repo via sparse git clone, RST extraction (PSF License)
- **Target:** 15,000 sentences
- **Description:** Technical and code-related discussion

### news

- **Sources:**
  - Wikinews -- articles via Wikinews API (CC BY 2.5)
  - Wikipedia -- news-related category articles via API (CC BY-SA 3.0)
- **Target:** 15,000 sentences
- **Description:** News articles with formal prose and complex sentences

### encyclopedic

- **Source:** Wikipedia general knowledge category articles via API
- **License:** CC BY-SA 3.0
- **Target:** 10,000 sentences
- **Description:** Factual, definitional content

## Full pipeline run

To build the entire corpus from scratch:

```bash
# 1. Collect all domains
uv run python -m corpus.domains.conversation.collect
uv run python -m corpus.domains.narrative.collect
uv run python -m corpus.domains.business.collect
uv run python -m corpus.domains.technical.collect
uv run python -m corpus.domains.news.collect
uv run python -m corpus.domains.encyclopedic.collect

# 2. Preprocess all domains
uv run python -m corpus.domains.conversation.preprocess
uv run python -m corpus.domains.narrative.preprocess
uv run python -m corpus.domains.business.preprocess
uv run python -m corpus.domains.technical.preprocess
uv run python -m corpus.domains.news.preprocess
uv run python -m corpus.domains.encyclopedic.preprocess

# 3. Annotate all domains
uv run python -m corpus.pipeline.annotate --domain conversation
uv run python -m corpus.pipeline.annotate --domain narrative
uv run python -m corpus.pipeline.annotate --domain business
uv run python -m corpus.pipeline.annotate --domain technical
uv run python -m corpus.pipeline.annotate --domain news
uv run python -m corpus.pipeline.annotate --domain encyclopedic

# 4. Validate all domains
uv run python -m corpus.pipeline.validate --domain conversation
uv run python -m corpus.pipeline.validate --domain narrative
uv run python -m corpus.pipeline.validate --domain business
uv run python -m corpus.pipeline.validate --domain technical
uv run python -m corpus.pipeline.validate --domain news
uv run python -m corpus.pipeline.validate --domain encyclopedic

# 5. Export final corpus
uv run python -m corpus.pipeline.export \
  --domains conversation narrative business technical news encyclopedic

# 6. Check stats
uv run python -m corpus.pipeline.stats --final
```
