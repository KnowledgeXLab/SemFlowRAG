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
export HF_HOME=/path/to/huggingface_home
export OPENAI_API_KEY=your_openai_api_key
```

For local vLLM serving, also set:

```sh
export VLLM_WORKER_MULTIPROC_METHOD=spawn
```

## Quick Start

Use the small `sample` dataset to verify the environment:

```sh

dataset=sample

# Use any OpenAI-compatible model name available to your API key.
python main.py \
  --dataset $dataset \
  --llm_base_url https://api.openai.com/v1 \
  --llm_name gpt-4o-mini \
  --embedding_name nvidia/NV-Embed-v2
```

## Local Deployment (vLLM)

Start an OpenAI-compatible vLLM server:

```sh
# Use 4 GPUs for vLLM serving.
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HOME=/path/to/huggingface_home

conda activate semflowrag

# Tune gpu-memory-utilization, max-model-len, tensor-parallel-size, and port to fit your environment.
model_name_or_path=meta-llama/Llama-3.3-70B-Instruct  # or a local model directory
served_model_name=Llama-3.3-70B-Instruct

python -m vllm.entrypoints.openai.api_server \
  --model $model_name_or_path \
  --port 8000 \
  --gpu-memory-utilization 0.90 \
  --tensor-parallel-size 4 \
  --max-model-len 16384 \
  --served-model-name $served_model_name \
  --trust-remote-code \
  --enable-chunked-prefill \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 256
```

In another terminal, run the experiment:

```sh

conda activate semflowrag

# Use 2 separate GPUs for the main pipeline, such as embedding computation.
export CUDA_VISIBLE_DEVICES=5,6
dataset=sample
llm_name=Llama-3.3-70B-Instruct

python main.py \
  --dataset $dataset \
  --llm_base_url http://localhost:8000/v1 \
  --llm_name $llm_name \
  --embedding_name nvidia/NV-Embed-v2
```


## Experimental Findings

The paper evaluates SemFlowRAG on NaturalQuestions, PopQA, MuSiQue, 2WikiMultiHopQA, HotpotQA, LV-Eval, and NarrativeQA. The main findings are:

- Directed semantic flow retrieves more complete evidence chains on multi-hop QA and reduces semantic drift from high-abstractness hub nodes.
- Compared with HippoRAG 2, GraphRAG, RAPTOR, LightRAG, BM25, Contriever, GTR, and NV-Embed-v2, SemFlowRAG achieves stronger average QA F1 and Recall@5.


# Acknowledgement
We gratefully acknowledge the use of the following open-source projects in our work:
[HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG)
