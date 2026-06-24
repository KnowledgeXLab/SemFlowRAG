import os
from dataclasses import dataclass, field
from typing import (
    Literal,
    Union,
    Optional
)

from .logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class BaseConfig:
    """One and only configuration."""
    # LLM specific attributes 
    llm_name: str = field(
        default="gpt-4o-mini",
        metadata={"help": "Class name indicating which LLM model to use."}
    )
    llm_base_url: str = field(
        default=None,
        metadata={"help": "Base URL for the LLM model, if none, means using OPENAI service."}
    )
    embedding_base_url: str = field(
        default=None,
        metadata={"help": "Base URL for an OpenAI compatible embedding model, if none, means using OPENAI service."}
    )
    azure_endpoint: str = field(
        default=None,
        metadata={"help": "Azure Endpoint URI for the LLM model, if none, uses OPENAI service directly."}
    )
    azure_embedding_endpoint: str = field(
        default=None,
        metadata={"help": "Azure Endpoint URI for the OpenAI embedding model, if none, uses OPENAI service directly."}
    )
    max_new_tokens: Union[None, int] = field(
        default=2048,
        metadata={"help": "Max new tokens to generate in each inference."}
    )
    num_gen_choices: int = field(
        default=1,
        metadata={"help": "How many chat completion choices to generate for each input message."}
    )
    seed: Union[None, int] = field(
        default=None,
        metadata={"help": "Random seed."}
    )
    temperature: float = field(
        default=0,
        metadata={"help": "Temperature for sampling in each inference."}
    )
    qa_temperature: Optional[float] = field(
        default=0.4,
        metadata={"help": "Temperature override for final QA generation only. If None, use the global temperature."}
    )
    response_format: Union[dict, None] = field(
        default_factory=lambda: { "type": "json_object" },
        metadata={"help": "Specifying the format that the model must output."}
    )
    
    ## LLM specific attributes -> Async hyperparameters
    max_retry_attempts: int = field(
        default=5,
        metadata={"help": "Max number of retry attempts for an asynchronous API calling."}
    )
    openie_ner_max_tokens: int = field(
        default=512,
        metadata={"help": "Max completion tokens for OpenIE NER calls."}
    )
    openie_triple_max_tokens: int = field(
        default=2048,
        metadata={"help": "Max completion tokens for OpenIE triple extraction calls."}
    )
    openie_num_workers: int = field(
        default=128,
        metadata={"help": "Max number of worker threads for online OpenIE NER, triple extraction, and retry calls."}
    )
    openie_retry_low_quality: bool = field(
        default=True,
        metadata={"help": "If True, retry sparse or failed OpenIE triple extraction outputs once with a stricter prompt."}
    )
    openie_retry_min_entities: int = field(
        default=4,
        metadata={"help": "Minimum number of named entities for sparse-triple OpenIE retry checks."}
    )
    openie_retry_min_triples: int = field(
        default=2,
        metadata={"help": "Retry triple extraction when a chunk has at least openie_retry_min_entities but fewer triples than this value."}
    )
    openie_retry_max_tokens: int = field(
        default=2048,
        metadata={"help": "Max completion tokens for low-quality OpenIE triple extraction retry calls."}
    )
    # Storage specific attributes
    force_openie_from_scratch: bool = field(
        default=False,
        metadata={"help": "If set to True, will ignore all existing openie files and rebuild them from scratch."}
    )

    # Storage specific attributes 
    force_index_from_scratch: bool = field(
        default=False,
        metadata={"help": "If set to True, will ignore all existing storage files and graph data and will rebuild from scratch."}
    )
    rerank_dspy_file_path: str = field(
        default=None,
        metadata={"help": "Path to the rerank dspy file."}
    )
    passage_node_weight: float = field(
        default=0.08,
        metadata={"help": "Multiplicative factor that modifies the passage node reset weights in PPR."}
    )
    save_openie: bool = field(
        default=True,
        metadata={"help": "If set to True, will save the OpenIE model to disk."}
    )
    
    # Preprocessing specific attributes
    text_preprocessor_class_name: str = field(
        default="TextPreprocessor",
        metadata={"help": "Name of the text-based preprocessor to use in preprocessing."}
    )
    preprocess_encoder_name: str = field(
        default="gpt-4o",
        metadata={"help": "Name of the encoder to use in preprocessing (currently implemented specifically for doc chunking)."}
    )
    preprocess_chunk_overlap_token_size: int = field(
        default=128,
        metadata={"help": "Number of overlap tokens between neighbouring chunks."}
    )
    preprocess_chunk_max_token_size: int = field(
        default=None,
        metadata={"help": "Max number of tokens each chunk can contain. If set to None, the whole doc will treated as a single chunk."}
    )
    preprocess_chunk_func: Literal["by_token", "by_word"] = field(default='by_token')
    
    
    # Information extraction specific attributes
    information_extraction_model_name: Literal["openie_openai_gpt", ] = field(
        default="openie_openai_gpt",
        metadata={"help": "Class name indicating which information extraction model to use."}
    )
    openie_mode: Literal["offline", "online"] = field(
        default="online",
        metadata={"help": "Mode of the OpenIE model to use."}
    )
    skip_graph: bool = field(
        default=False,
        metadata={"help": "Whether to skip graph construction or not. Set it to be true when running vllm offline indexing for the first time."}
    )
    
    
    # Embedding specific attributes
    embedding_model_name: str = field(
        default="nvidia/NV-Embed-v2",
        metadata={"help": "Class name indicating which embedding model to use."}
    )
    embedding_batch_size: int = field(
        default=16,
        metadata={"help": "Batch size of calling embedding model."}
    )
    embedding_return_as_normalized: bool = field(
        default=True,
        metadata={"help": "Whether to normalize encoded embeddings not."}
    )
    embedding_max_seq_len: int = field(
        default=2048,
        metadata={"help": "Max sequence length for the embedding model."}
    )
    embedding_model_dtype: Literal["float16", "float32", "bfloat16", "auto"] = field(
        default="auto",
        metadata={"help": "Data type for local embedding model."}
    )
    
    
    
    # Graph construction specific attributes
    synonymy_edge_topk: int = field(
        default=2047,
        metadata={"help": "k for knn retrieval in buiding synonymy edges."}
    )
    synonymy_edge_query_batch_size: int = field(
        default=1000,
        metadata={"help": "Batch size for query embeddings for knn retrieval in buiding synonymy edges."}
    )
    synonymy_edge_key_batch_size: int = field(
        default=10000,
        metadata={"help": "Batch size for key embeddings for knn retrieval in buiding synonymy edges."}
    )
    synonymy_edge_sim_threshold: float = field(
        default=0.8,
        metadata={"help": "Similarity threshold to include candidate synonymy nodes."}
    )
    is_directed_graph: bool = field(
        default=False,
        metadata={"help": "Whether the graph is directed or not."}
    )
    
    
    
    # Retrieval specific attributes
    linking_top_k: int = field(
        default=5,
        metadata={"help": "The number of linked nodes at each retrieval step"}
    )
    entity_seed_top_k: int = field(
        default=5,
        metadata={"help": "The number of entity seed nodes retained for the PPR reset after fact reranking."}
    )
    skip_fact_rerank: bool = field(
        default=False,
        metadata={"help": "If True, skip the LLM/DSPy fact filter and use embedding top-k facts directly."}
    )
    retrieval_top_k: int = field(
        default=200,
        metadata={"help": "Retrieving k documents at each step"}
    )
    retrieval_num_workers: int = field(
        default=16,
        metadata={"help": "Number of parallel workers for query retrieval. Set 1 for fully serial behavior."}
    )
    damping: float = field(
        default=0.5,
        metadata={"help": "Damping factor for ppr algorithm."}
    )

    # Directed PPR with diversity-guided walk (Innovation)
    enable_directed_ppr: bool = field(
        default=False,
        metadata={"help": "Master switch for diversity-guided directed PPR. When True, graph is built directed and edge weights are rewritten per-query with a hard 9:1 (down:up) within-direction split."}
    )
    ppr_down_direction_prob: float = field(
        default=0.9,
        metadata={"help": "Within-row probability budget allocated to edges where delta_div <= 0 (down direction). With damping=0.5 this yields per-step absolute probability 0.45."}
    )
    ppr_up_direction_prob: float = field(
        default=0.1,
        metadata={"help": "Within-row probability budget allocated to edges where delta_div > 0 (up direction). With damping=0.5 this yields per-step absolute probability 0.05."}
    )
    ppr_score_lambda_rel: float = field(
        default=1.0,
        metadata={"help": "Weight on normalized per-entity relevance in the edge score."}
    )
    ppr_score_const_base: float = field(
        default=0.0,
        metadata={"help": "Constant base term added to the directed-PPR edge score. Keep 0.0 for the main method; set 1.0 for no-query-relevance ablation score=1-|delta_div|."}
    )
    ppr_use_prior_edge_weight: bool = field(
        default=False,
        metadata={"help": "Whether to include normalized fact/synonymy edge prior in directed-PPR edge scoring. When False, fact/synonymy edges only provide topology."}
    )
    ppr_score_lambda_prior: float = field(
        default=1.0,
        metadata={"help": "Weight on normalized synonymy/fact prior in the edge score."}
    )
    ppr_score_lambda_div: float = field(
        default=1.0,
        metadata={"help": "Weight on |delta_div| (subtracted) in the edge score, controlling how strongly the walk prefers small per-step diversity changes."}
    )
    ppr_score_epsilon: float = field(
        default=1e-6,
        metadata={"help": "Floor for per-edge score after ReLU non-negativization, so an entirely zeroed direction bucket is rare and self-loop / forced-direction fallback does not dominate mass flow."}
    )
    ppr_truncation_low: float = field(
        default=0.01,
        metadata={"help": "Lower quantile for truncated Min-Max normalization of rel / div / prior."}
    )
    ppr_truncation_high: float = field(
        default=0.99,
        metadata={"help": "Upper quantile for truncated Min-Max normalization of rel / div / prior."}
    )
    ppr_self_loop_at_local_min: bool = field(
        default=True,
        metadata={"help": "A+ option: when a node has no down-direction neighbors (S_D=0), absorb mass via a self-loop edge carrying the down-direction budget. When False, merge the down budget into the up direction instead (pure option A)."}
    )
    ppr_skip_direction_control: bool = field(
        default=False,
        metadata={"help": "Ablation flag. When True, the score formula (lambda_rel*rel - lambda_div*|delta_div|) is still computed per entity-entity edge, but the 9:1 directional bucketing, within-direction normalization, and A+ self-loop absorption are all skipped. The non-negativized score is used as the raw directed edge weight. Tests whether the direction-control mechanism (independent of the edge-score formula) is necessary."}
    )

    # Retrieval artifact dump for offline analysis (seed-vs-topk graph proximity)
    dump_retrieval_artifacts: bool = field(
        default=False,
        metadata={"help": "If True, append one JSONL line per query into {save_dir}/retrieval_artifacts.jsonl with entity seeds, DPR passage seeds, and top-K PPR passages for post-hoc analysis."}
    )
    retrieval_artifacts_top_k_passage: int = field(
        default=100,
        metadata={"help": "Number of top PPR passages to record per query in the retrieval artifact JSONL."}
    )
    retrieval_artifacts_top_k_seed: int = field(
        default=5,
        metadata={"help": "Number of top DPR passage seeds (by raw DPR score) to record per query in the retrieval artifact JSONL. Entity seeds always follow entity_seed_top_k."}
    )

    # Probability black-hole experiment: record top-K entity nodes by converged PPR
    # score together with their diversity & query-similarity, so we can analyze
    # whether high-diversity / low-similarity hub entities absorb most of the mass.
    dump_entity_black_hole: bool = field(
        default=False,
        metadata={"help": "If True, append one JSONL line per query into {save_dir}/entity_black_hole_artifacts.jsonl with the top-K entity nodes by PPR score together with their diversity & query-similarity."}
    )
    entity_black_hole_top_k: int = field(
        default=100,
        metadata={"help": "Number of top entity nodes by PPR converged score to record per query."}
    )
    sample_random_seed: Optional[int] = field(
        default=None,
        metadata={"help": "If set, draw `sample_size` queries via random.sample with this seed instead of taking the first N. SemFlowRAG class itself does not consume this field."}
    )

    # Innovation vs Vanilla PPR comparison experiment on entity-only subgraph.
    # Side-channel only: main retrieval pipeline is unaffected.
    ppr_experiment_mode: str = field(
        default="none",
        metadata={"help": "Side-channel PPR experiment. One of: 'none' (default, off), 'entity_only_innovation' (Innovation directed-PPR edge rewrite on entity-only subgraph), 'entity_only_vanilla' (vanilla PPR on entity-only subgraph with default graph.es['weight'])."}
    )


    # QA specific attributes
    max_qa_steps: int = field(
        default=1,
        metadata={"help": "For answering a single question, the max steps that we use to interleave retrieval and reasoning."}
    )
    qa_top_k: int = field(
        default=5,
        metadata={"help": "Feeding top k documents to the QA model for reading."}
    )
    qa_num_workers: int = field(
        default=32,
        metadata={"help": "Number of parallel workers for QA LLM calls. Set 1 for fully serial behavior."}
    )
    qa_max_new_tokens: Optional[int] = field(
        default=None,
        metadata={"help": "Max completion tokens for final QA generation. If None, follows the baseline/default LLM behavior."}
    )
    
    # Save dir (highest level directory)
    save_dir: str = field(
        default=None,
        metadata={"help": "Directory to save all related information. If it's given, will overwrite all default save_dir setups. If it's not given, then if we're not running specific datasets, default to `outputs`, otherwise, default to a dataset-customized output dir."}
    )
    
    
    
    # Dataset running specific attributes
    ## Dataset running specific attributes -> General
    dataset: Optional[Literal['hotpotqa', 'hotpotqa_train', 'musique', '2wikimultihopqa']] = field(
        default=None,
        metadata={"help": "Dataset to use. If specified, it means we will run specific datasets. If not specified, it means we're running freely."}
    )
    ## Dataset running specific attributes -> Graph
    graph_type: Literal[
        'dpr_only', 
        'entity', 
    ] = field(
        default="entity",
        metadata={"help": "Type of graph to use in the experiment."}
    )
    corpus_len: Optional[int] = field(
        default=None,
        metadata={"help": "Length of the corpus to use."}
    )
    
    
    def __post_init__(self):
        if self.save_dir is None: # If save_dir not given
            if self.dataset is None: self.save_dir = 'outputs' # running freely
            else: self.save_dir = os.path.join('outputs', self.dataset) # customize your dataset's output dir here
        logger.debug(f"Initializing the highest level of save_dir to be {self.save_dir}")

        if self.enable_directed_ppr and not self.is_directed_graph:
            logger.info("enable_directed_ppr=True forces is_directed_graph=True.")
            self.is_directed_graph = True
