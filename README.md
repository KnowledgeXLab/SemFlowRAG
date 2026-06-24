# SemFlowRAG: Directed Semantic Flow from Abstraction to Evidence for Complex Reasoning


SemFlowRAG is a novel framework for complex reasoning that uses a Semantic Gradient Knowledge Graph to organize retrieval. Through directed semantic flow, it guides search from abstract concepts to concrete evidence. This hierarchical approach ensures generated responses are logically rigorous and grounded in precise factual details.


## Method Pipeline

<p align="center">
  <img src="images/methodology.png" width="90%" alt="SemFlowRAG offline semantic gradient indexing and online directed PPR search">
</p>


## Installation

```sh

conda create -n semflowrag python=3.10
conda activate semflowrag

pip install -e .
```

Initialize the environmental variables:

```sh
export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_HOME=<your huggingface home>
export OPENAI_API_KEY=<your openai api key>
```

For local vLLM serving, also set:

```sh
export VLLM_WORKER_MULTIPROC_METHOD=spawn
```

## Quick Start

Use the small `sample` dataset to verify the environment:

```sh

dataset=sample
python main.py \
  --dataset $dataset \
  --llm_base_url https://api.openai.com/v1 \
  --llm_name gpt-4o-mini \ # Any OpenAI model name
  --embedding_name nvidia/NV-Embed-v2
```

## Local vLLM Usage

Start an OpenAI-compatible vLLM server:

```sh
export CUDA_VISIBLE_DEVICES=0,1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HOME=<your huggingface home>

conda activate semflowrag
vllm serve meta-llama/Llama-3.3-70B-Instruct \
  --tensor-parallel-size 2 \
  --max_model_len 4096 \
  --gpu-memory-utilization 0.95
```

In another terminal, run the experiment:

```sh

conda activate semflowrag

export CUDA_VISIBLE_DEVICES=2,3
dataset=sample

python main.py \
  --dataset $dataset \
  --llm_base_url http://localhost:8000/v1 \
  --llm_name meta-llama/Llama-3.3-70B-Instruct \
  --embedding_name nvidia/NV-Embed-v2
```

If GPU memory is insufficient, reduce `--max_model_len`, reduce `--gpu-memory-utilization`, or reserve separate GPUs for the embedding model.

## OpenIE Offline Batch Mode

For larger corpora, online OpenIE can be slow. You can first generate OpenIE caches with vLLM offline batch mode:

```sh

conda activate semflowrag

export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HOME=<your_huggingface_home>
export OPENAI_API_KEY=""

dataset=sample
python main.py \
  --dataset $dataset \
  --llm_name meta-llama/Llama-3.3-70B-Instruct \
  --embedding_name nvidia/NV-Embed-v2 \
  --openie_mode offline
```

The offline step writes OpenIE results. Then start a vLLM online server and rerun `main.py` with the same `dataset`, `llm_name`, and `embedding_name`; the program will reuse the OpenIE cache and continue with graph construction, retrieval, and QA.

## Custom Datasets

Place custom data under `reproduce/dataset/` and keep paired names:

- `{name}_corpus.json`: retrieval corpus.
- `{name}.json`: queries, answers, and optional gold evidence.

Corpus format:

```json
[
  {
    "title": "FIRST PASSAGE TITLE",
    "text": "FIRST PASSAGE TEXT",
    "idx": 0
  },
  {
    "title": "SECOND PASSAGE TITLE",
    "text": "SECOND PASSAGE TEXT",
    "idx": 1
  }
]
```

Query format:

```json
[
  {
    "id": "sample/question_1.json",
    "question": "QUESTION",
    "answer": ["ANSWER"],
    "paragraphs": [
      {
        "title": "SUPPORTING PASSAGE TITLE",
        "text": "SUPPORTING PASSAGE TEXT",
        "is_supporting": true,
        "idx": 0
      }
    ]
  }
]
```

`main.py` supports common evidence schemas such as `supporting_facts`, `contexts`, and `paragraphs`. If no gold evidence is available, QA can still run, but retrieval recall cannot be evaluated.

## Outputs and Caches

The default output directory is `outputs/{dataset}`. Common files include:

- `openie_results_ner_{llm}.json`: OpenIE extraction cache.
- `{llm_name}_{embedding_name}/chunk_embeddings/`: passage embedding store.
- `{llm_name}_{embedding_name}/entity_embeddings/`: entity embedding store.
- `{llm_name}_{embedding_name}/fact_embeddings/`: fact embedding store.
- `{llm_name}_{embedding_name}/graph.pickle`: constructed graph.
- `retrieval_artifacts.jsonl`: generated when `dump_retrieval_artifacts=True`.
- `entity_black_hole_artifacts.jsonl`: generated when `dump_entity_black_hole=True`.

To rerun an experiment from scratch, remove both the OpenIE cache and the model-specific working directory, for example:

```sh
rm outputs/sample/openie_results_ner_meta-llama_Llama-3.3-70B-Instruct.json
rm -rf outputs/sample/meta-llama_Llama-3.3-70B-Instruct_nvidia_NV-Embed-v2
```

Actual directory names are derived from `llm_name` and `embedding_name` by replacing `/` with `_`; check the runtime logs if unsure.

## Tests

OpenAI-compatible test:

```sh

conda activate semflowrag
export OPENAI_API_KEY=<your_openai_api_key>

python tests_openai.py
```

Local vLLM test:

```sh

conda activate semflowrag

export CUDA_VISIBLE_DEVICES=1
python tests_local.py
```

## Experimental Findings

The paper evaluates SemFlowRAG on NaturalQuestions, PopQA, MuSiQue, 2WikiMultiHopQA, HotpotQA, LV-Eval, and NarrativeQA. The main findings are:

- Directed semantic flow retrieves more complete evidence chains on multi-hop QA and reduces semantic drift from high-abstractness hub nodes.
- Compared with HippoRAG 2, GraphRAG, RAPTOR, LightRAG, BM25, Contriever, GTR, and NV-Embed-v2, SemFlowRAG achieves stronger average QA F1 and Recall@5.
- Ablations show that direction control, query relevance, and the abstractness penalty all contribute to performance.

See the paper PDF in the repository root for full experimental settings, numeric results, ablations, and the case study.

## Notes

- `enable_directed_ppr` is not currently exposed as a CLI argument. To reproduce the main paper method, set it explicitly in `BaseConfig`.
- `max_qa_steps` exists in the config, but the current main QA path reads top-k evidence once and generates the answer in a single call; it is not a default multi-step IRCoT loop.
- OpenIE quality directly affects graph quality. Missing entities, incorrect triples, or noisy corpora can degrade abstractness estimation and retrieval.
- Directed semantic flow assumes that most complex questions benefit from moving from abstract concepts toward concrete evidence. For tasks requiring lateral association or bottom-up inference, tune the direction ratio, reset probability, or ablation flags.
- Large LLMs and embedding models require substantial GPU memory. Start with the `sample` dataset before scaling to full benchmarks.
