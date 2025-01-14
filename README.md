<div align="center">
  <img src="https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo-pirate.png" width="200px" alt="smol blueprint logo">
</div>

# A smol blueprint project

A smol blueprint project showing a practical example of scraping, RAG, information extraction, analysis and fine-tuning in the age of LLMs. It is a more practical project that strives to apply    some of the learning from [the smol-course](https://github.com/huggingface/smol-course) to an end2end real-world problem.

> 🚀 Ready for production!
>
> Each notebook will show how to deploy your AI tools as an interactivedemo on Hugging Face Spaces with Gradio, which you can directly use as microservices through [the Gradio Python Client](https://www.gradio.app/guides/getting-started-with-the-python-client).

## The problem

We want to build a tool that can help us answer questions about the Hugging Face ecosystem. In reality, this data is nicely structured and available through [Hub API endpoints](https://huggingface.co/docs/hub/en/api), however, we will assume we need to gather data from the Hub blogs website.

-  [WIP] Managing a RAG pipeline.
  - [.ipynb](./rag/indexing.ipynb) - the Hugging Face Hub as a vector search backend.
  - Creating a RAG pipeline.
  - Deploying the RAG pipeline to a Hugging Face Space.
  - Monitoring the RAG pipeline.
  - Fine-tuning the retrieval and reranking models.
- [on hold 🛑] Information extraction: Structured information extraction with LLMs.
  - Extracting structured information from the blogs.
  - Deploying the information extraction pipeline to a Hugging Face Space.
  - Monitoring the information extraction pipeline.
  - Fine-tuning the information extraction models.
- [on hold 🛑] Agents: Orchestrate interactions with the other components.

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

You will need a Hugging Face account to use the Hub API. You can create one [here](https://huggingface.co/join). After this you can follow the [huggingface-cli instructions](https://huggingface.co/docs/huggingface_hub/installation#huggingface-cli) and log in to configure your Hugging Face token.

```bash
huggingface-cli login
```

