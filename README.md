# a smol blueprint project

A smol blueprint project showing a practical example scraping, RAG, information extraction, analysis and fine-tuning in the age of LLMs.

This is a practical project that strives to apply some of the learning from [the smol-course](https://github.com/huggingface/smol-course) to an end2end real-world problem.

## The problem

We want to build a tool that can help us answer questions about the Hugging Face ecosystem. In reality, this data is nicely structured and available through [Hub API endpoints](https://huggingface.co/docs/hub/en/api), however, we will assume we need to gather data from the Hub blogs website.

- Scraping: LLM guided web scraping.
- RAG: Indexing and optimizing a RAG pipeline.
- Information extraction: Structured information extraction.
- Fine-tuning: Fine-tuning predictive and generative models.

# Installation and configuration

## Python environment

We will use [uv](https://docs.astral.sh/uv/) to manage the project. First create a virtual environment:

```bash
uv venv --python 3.11
source .venv/bin/activate
```

Then you can install all the required dependencies:

```bash
uv sync --all-groups
```

Or you can sync between different dependency groups:

```bash
uv sync scraping
uv sync rag
uv sync information-extraction
```

## Hugging Face Account

You will need a Hugging Face account to use the Hub API. You can create one [here](https://huggingface.co/join). After this you can follow the [huggingface-cli instructions](https://huggingface.co/docs/huggingface_hub/installation#huggingface-cli) and log in to configure your token.

```bash
huggingface-cli login
```

