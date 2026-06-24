import json
import os
import logging
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Union, Optional, List, Set, Dict, Any, Tuple, Literal
import numpy as np
import importlib
from collections import defaultdict
from transformers import HfArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from igraph import Graph
import igraph as ig
import numpy as np
from collections import defaultdict
import re
import time

from .llm import _get_llm_class, BaseLLM
from .embedding_model import _get_embedding_model_class, BaseEmbeddingModel
from .embedding_store import EmbeddingStore
from .information_extraction import OpenIE
from .information_extraction.openie_vllm_offline import VLLMOfflineOpenIE
from .information_extraction.openie_transformers_offline import TransformersOfflineOpenIE
from .evaluation.retrieval_eval import RetrievalRecall
from .evaluation.qa_eval import QAExactMatch, QAF1Score
from .prompts.linking import get_query_instruction
from .prompts.prompt_template_manager import PromptTemplateManager
from .rerank import DSPyFilter
from .utils.misc_utils import *
from .utils.misc_utils import NerRawOutput, TripleRawOutput
from .utils.embed_utils import retrieve_knn
from .utils.typing import Triple
from .utils.config_utils import BaseConfig

logger = logging.getLogger(__name__)

class SemFlowRAG:

    def __init__(self,
                 global_config=None,
                 save_dir=None,
                 llm_model_name=None,
                 llm_base_url=None,
                 embedding_model_name=None,
                 embedding_base_url=None,
                 azure_endpoint=None,
                 azure_embedding_endpoint=None):
        """
        Initializes an instance of the class and its related components.

        Attributes:
            global_config (BaseConfig): The global configuration settings for the instance. An instance
                of BaseConfig is used if no value is provided.
            saving_dir (str): The directory where specific SemFlowRAG instances will be stored. This defaults
                to `outputs` if no value is provided.
            llm_model (BaseLLM): The language model used for processing based on the global
                configuration settings.
            openie (Union[OpenIE, VLLMOfflineOpenIE]): The Open Information Extraction module
                configured in either online or offline mode based on the global settings.
            graph: The graph instance initialized by the `initialize_graph` method.
            embedding_model (BaseEmbeddingModel): The embedding model associated with the current
                configuration.
            chunk_embedding_store (EmbeddingStore): The embedding store handling chunk embeddings.
            entity_embedding_store (EmbeddingStore): The embedding store handling entity embeddings.
            fact_embedding_store (EmbeddingStore): The embedding store handling fact embeddings.
            prompt_template_manager (PromptTemplateManager): The manager for handling prompt templates
                and roles mappings.
            openie_results_path (str): The file path for storing Open Information Extraction results
                based on the dataset and LLM name in the global configuration.
            rerank_filter (Optional[DSPyFilter]): The filter responsible for reranking information
                when a rerank file path is specified in the global configuration.
            ready_to_retrieve (bool): A flag indicating whether the system is ready for retrieval
                operations.

        Parameters:
            global_config: The global configuration object. Defaults to None, leading to initialization
                of a new BaseConfig object.
            working_dir: The directory for storing working files. Defaults to None, constructing a default
                directory based on the class name and timestamp.
            llm_model_name: LLM model name, can be inserted directly as well as through configuration file.
            embedding_model_name: Embedding model name, can be inserted directly as well as through configuration file.
            llm_base_url: LLM URL for a deployed LLM model, can be inserted directly as well as through configuration file.
        """
        if global_config is None:
            self.global_config = BaseConfig()
        else:
            self.global_config = global_config

        #Overwriting Configuration if Specified
        if save_dir is not None:
            self.global_config.save_dir = save_dir

        if llm_model_name is not None:
            self.global_config.llm_name = llm_model_name

        if embedding_model_name is not None:
            self.global_config.embedding_model_name = embedding_model_name

        if llm_base_url is not None:
            self.global_config.llm_base_url = llm_base_url

        if embedding_base_url is not None:
            self.global_config.embedding_base_url = embedding_base_url

        if azure_endpoint is not None:
            self.global_config.azure_endpoint = azure_endpoint

        if azure_embedding_endpoint is not None:
            self.global_config.azure_embedding_endpoint = azure_embedding_endpoint

        _print_config = ",\n  ".join([f"{k} = {v}" for k, v in asdict(self.global_config).items()])
        logger.debug(f"SemFlowRAG init with config:\n  {_print_config}\n")

        #LLM and embedding model specific working directories are created under every specified saving directories
        llm_label = self.global_config.llm_name.replace("/", "_")
        embedding_label = self.global_config.embedding_model_name.replace("/", "_")
        self.working_dir = os.path.join(self.global_config.save_dir, f"{llm_label}_{embedding_label}")

        if not os.path.exists(self.working_dir):
            logger.info(f"Creating working directory: {self.working_dir}")
            os.makedirs(self.working_dir, exist_ok=True)

        self.llm_model: BaseLLM = _get_llm_class(self.global_config)

        if self.global_config.openie_mode == 'online':
            self.openie = OpenIE(llm_model=self.llm_model)
        elif self.global_config.openie_mode == 'offline':
            self.openie = VLLMOfflineOpenIE(self.global_config)
        elif self.global_config.openie_mode ==  'Transformers-offline':
            self.openie = TransformersOfflineOpenIE(self.global_config)

        self.graph = self.initialize_graph()

        if self.global_config.openie_mode == 'offline':
            self.embedding_model = None
        else:
            self.embedding_model: BaseEmbeddingModel = _get_embedding_model_class(
                embedding_model_name=self.global_config.embedding_model_name)(global_config=self.global_config,
                                                                              embedding_model_name=self.global_config.embedding_model_name)
        self.chunk_embedding_store = EmbeddingStore(self.embedding_model,
                                                    os.path.join(self.working_dir, "chunk_embeddings"),
                                                    self.global_config.embedding_batch_size, 'chunk')
        self.entity_embedding_store = EmbeddingStore(self.embedding_model,
                                                     os.path.join(self.working_dir, "entity_embeddings"),
                                                     self.global_config.embedding_batch_size, 'entity')
        self.fact_embedding_store = EmbeddingStore(self.embedding_model,
                                                   os.path.join(self.working_dir, "fact_embeddings"),
                                                   self.global_config.embedding_batch_size, 'fact')

        self.prompt_template_manager = PromptTemplateManager(role_mapping={"system": "system", "user": "user", "assistant": "assistant"})

        self.openie_results_path = os.path.join(self.global_config.save_dir,f'openie_results_ner_{self.global_config.llm_name.replace("/", "_")}.json')

        self.rerank_filter = DSPyFilter(self)

        self.ready_to_retrieve = False

        self.ppr_time = 0
        self.rerank_time = 0
        self.all_retrieval_time = 0
        self._timing_lock = threading.Lock()
        self._artifact_lock = threading.Lock()

        self.ent_node_to_chunk_ids = None


    def initialize_graph(self):
        """
        Initializes a graph using a Pickle file if available or creates a new graph.

        The function attempts to load a pre-existing graph stored in a Pickle file. If the file
        is not present or the graph needs to be created from scratch, it initializes a new directed
        or undirected graph based on the global configuration. If the graph is loaded successfully
        from the file, pertinent information about the graph (number of nodes and edges) is logged.

        Returns:
            ig.Graph: A pre-loaded or newly initialized graph.

        Raises:
            None
        """
        self._graph_pickle_filename = os.path.join(
            self.working_dir, f"graph.pickle"
        )

        preloaded_graph = None

        if not self.global_config.force_index_from_scratch:
            if os.path.exists(self._graph_pickle_filename):
                preloaded_graph = ig.Graph.Read_Pickle(self._graph_pickle_filename)

        if preloaded_graph is None:
            return ig.Graph(directed=self.global_config.is_directed_graph)
        else:
            logger.info(
                f"Loaded graph from {self._graph_pickle_filename} with {preloaded_graph.vcount()} nodes, {preloaded_graph.ecount()} edges"
            )
            return preloaded_graph

    def pre_openie(self,  docs: List[str]):
        logger.info(f"Indexing Documents")
        logger.info(f"Performing OpenIE Offline")

        chunks = self.chunk_embedding_store.get_missing_string_hash_ids(docs)

        all_openie_info, chunk_keys_to_process = self.load_existing_openie(chunks.keys())
        new_openie_rows = {k : chunks[k] for k in chunk_keys_to_process}

        if len(chunk_keys_to_process) > 0:
            new_ner_results_dict, new_triple_results_dict = self.openie.batch_openie(new_openie_rows)
            self.merge_openie_results(all_openie_info, new_openie_rows, new_ner_results_dict, new_triple_results_dict)

        if self.global_config.save_openie:
            self.save_openie_results(all_openie_info)

        assert False, logger.info('Done with OpenIE, run online indexing for future retrieval.')

    def index(self, docs: List[str]):
        """
        Indexes the given documents based on the SemFlowRAG 2 framework which generates an OpenIE knowledge graph
        based on the given documents and encodes passages, entities and facts separately for later retrieval.

        Parameters:
            docs : List[str]
                A list of documents to be indexed.
        """

        logger.info(f"Indexing Documents")

        logger.info(f"Performing OpenIE")

        if self.global_config.openie_mode == 'offline':
            self.pre_openie(docs)

        self.chunk_embedding_store.insert_strings(docs)
        chunk_to_rows = self.chunk_embedding_store.get_all_id_to_rows()

        all_openie_info, chunk_keys_to_process = self.load_existing_openie(chunk_to_rows.keys())
        new_openie_rows = {k : chunk_to_rows[k] for k in chunk_keys_to_process}

        if len(chunk_keys_to_process) > 0:
            new_ner_results_dict, new_triple_results_dict = self.openie.batch_openie(new_openie_rows)
            self.merge_openie_results(all_openie_info, new_openie_rows, new_ner_results_dict, new_triple_results_dict)

        if self.global_config.save_openie:
            self.save_openie_results(all_openie_info)

        ner_results_dict, triple_results_dict = reformat_openie_results(all_openie_info)

        assert len(chunk_to_rows) == len(ner_results_dict) == len(triple_results_dict), f"len(chunk_to_rows): {len(chunk_to_rows)}, len(ner_results_dict): {len(ner_results_dict)}, len(triple_results_dict): {len(triple_results_dict)}"

        # prepare data_store
        chunk_ids = list(chunk_to_rows.keys())

        chunk_triples = [[text_processing(t) for t in triple_results_dict[chunk_id].triples] for chunk_id in chunk_ids]
        entity_nodes, chunk_triple_entities = extract_entity_nodes(chunk_triples)
        facts = flatten_facts(chunk_triples)

        logger.info(f"Encoding Entities")
        self.entity_embedding_store.insert_strings(entity_nodes)

        logger.info(f"Encoding Facts")
        self.fact_embedding_store.insert_strings([str(fact) for fact in facts])

        logger.info(f"Constructing Graph")

        self.node_to_node_stats = {}
        self.ent_node_to_chunk_ids = {}

        self.add_fact_edges(chunk_ids, chunk_triples)
        self.add_passage_edges(chunk_ids, chunk_triple_entities)

        if len(chunk_keys_to_process) > 0 or self.graph.vcount() == 0:
            if len(chunk_keys_to_process) > 0:
                logger.info(f"Found {len(chunk_keys_to_process)} new chunks to save into graph.")
            else:
                logger.info("Graph is empty; rebuilding graph from existing OpenIE results.")
            self.add_synonymy_edges()

            self.augment_graph()

            if self.global_config.enable_directed_ppr:
                self._ensure_self_loops()
                self.precompute_prior_norm()
                self.precompute_entity_diversity()

            self.save_igraph()

    def delete(self, docs_to_delete: List[str]):
        """
        Deletes the given documents from all data structures within the SemFlowRAG class.
        Note that triples and entities which are indexed from chunks that are not being removed will not be removed.

        Parameters:
            docs : List[str]
                A list of documents to be deleted.
        """

        #Making sure that all the necessary structures have been built.
        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()

        current_docs = set(self.chunk_embedding_store.get_all_texts())
        docs_to_delete = [doc for doc in docs_to_delete if doc in current_docs]

        #Get ids for chunks to delete
        chunk_ids_to_delete = set(
            [self.chunk_embedding_store.text_to_hash_id[chunk] for chunk in docs_to_delete])

        #Find triples in chunks to delete
        all_openie_info, chunk_keys_to_process = self.load_existing_openie([])
        triples_to_delete = []

        all_openie_info_with_deletes = []

        for openie_doc in all_openie_info:
            if openie_doc['idx'] in chunk_ids_to_delete:
                triples_to_delete.append(openie_doc['extracted_triples'])
            else:
                all_openie_info_with_deletes.append(openie_doc)

        triples_to_delete = flatten_facts(triples_to_delete)

        #Filter out triples that appear in unaltered chunks
        true_triples_to_delete = []

        for triple in triples_to_delete:
            proc_triple = tuple(text_processing(list(triple)))

            doc_ids = self.proc_triples_to_docs[str(proc_triple)]

            non_deleted_docs = doc_ids.difference(chunk_ids_to_delete)

            if len(non_deleted_docs) == 0:
                true_triples_to_delete.append(triple)

        processed_true_triples_to_delete = [[text_processing(list(triple)) for triple in true_triples_to_delete]]
        entities_to_delete, _ = extract_entity_nodes(processed_true_triples_to_delete)
        processed_true_triples_to_delete = flatten_facts(processed_true_triples_to_delete)

        triple_ids_to_delete = set([self.fact_embedding_store.text_to_hash_id[str(triple)] for triple in processed_true_triples_to_delete])

        #Filter out entities that appear in unaltered chunks
        ent_ids_to_delete = [self.entity_embedding_store.text_to_hash_id[ent] for ent in entities_to_delete]

        filtered_ent_ids_to_delete = []

        for ent_node in ent_ids_to_delete:
            doc_ids = self.ent_node_to_chunk_ids[ent_node]

            non_deleted_docs = doc_ids.difference(chunk_ids_to_delete)

            if len(non_deleted_docs) == 0:
                filtered_ent_ids_to_delete.append(ent_node)

        logger.info(f"Deleting {len(chunk_ids_to_delete)} Chunks")
        logger.info(f"Deleting {len(triple_ids_to_delete)} Triples")
        logger.info(f"Deleting {len(filtered_ent_ids_to_delete)} Entities")

        self.save_openie_results(all_openie_info_with_deletes)

        self.entity_embedding_store.delete(filtered_ent_ids_to_delete)
        self.fact_embedding_store.delete(triple_ids_to_delete)
        self.chunk_embedding_store.delete(chunk_ids_to_delete)

        #Delete Nodes from Graph
        self.graph.delete_vertices(list(filtered_ent_ids_to_delete))
        self.save_igraph()

        self.ready_to_retrieve = False

    def retrieve(self,
                 queries: List[str],
                 num_to_retrieve: int = None,
                 gold_docs: List[List[str]] = None) -> List[QuerySolution] | Tuple[List[QuerySolution], Dict]:
        """
        Performs retrieval using the SemFlowRAG 2 framework, which consists of several steps:
        - Fact Retrieval
        - Recognition Memory for improved fact selection
        - Dense passage scoring
        - Personalized PageRank based re-ranking

        Parameters:
            queries: List[str]
                A list of query strings for which documents are to be retrieved.
            num_to_retrieve: int, optional
                The maximum number of documents to retrieve for each query. If not specified, defaults to
                the `retrieval_top_k` value defined in the global configuration.
            gold_docs: List[List[str]], optional
                A list of lists containing gold-standard documents corresponding to each query. Required
                if retrieval performance evaluation is enabled (`do_eval_retrieval` in global configuration).

        Returns:
            List[QuerySolution] or (List[QuerySolution], Dict)
                If retrieval performance evaluation is not enabled, returns a list of QuerySolution objects, each containing
                the retrieved documents and their scores for the corresponding query. If evaluation is enabled, also returns
                a dictionary containing the evaluation metrics computed over the retrieved results.

        Notes
        -----
        - Long queries with no relevant facts after reranking will default to results from dense passage retrieval.
        """
        retrieve_start_time = time.time()  # Record start time

        if num_to_retrieve is None:
            num_to_retrieve = self.global_config.retrieval_top_k

        if gold_docs is not None:
            retrieval_recall_evaluator = RetrievalRecall(global_config=self.global_config)

        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()

        self.get_query_embeddings(queries)

        def retrieve_one(q_idx: int, query: str) -> Tuple[int, QuerySolution, float]:
            rerank_start = time.time()
            query_fact_scores = self.get_fact_scores(query)
            top_k_fact_indices, top_k_facts, rerank_log = self.rerank_facts(query, query_fact_scores)
            rerank_end = time.time()

            if len(top_k_facts) == 0:
                logger.info('No facts found after reranking, return DPR results')
                sorted_doc_ids, sorted_doc_scores = self.dense_passage_retrieval(query)
            else:
                sorted_doc_ids, sorted_doc_scores = self.graph_search_with_fact_entities(query=query,
                                                                                         link_top_k=self.global_config.linking_top_k,
                                                                                         query_fact_scores=query_fact_scores,
                                                                                         top_k_facts=top_k_facts,
                                                                                         top_k_fact_indices=top_k_fact_indices,
                                                                                         entity_seed_top_k=self.global_config.entity_seed_top_k,
                                                                                         passage_node_weight=self.global_config.passage_node_weight,
                                                                                         query_idx=q_idx)

            top_k_docs = [self.chunk_embedding_store.get_row(self.chunk_keys[idx])["content"] for idx in sorted_doc_ids[:num_to_retrieve]]

            return q_idx, QuerySolution(question=query, docs=top_k_docs, doc_scores=sorted_doc_scores[:num_to_retrieve]), rerank_end - rerank_start

        retrieval_workers = max(1, int(getattr(self.global_config, "retrieval_num_workers", 1) or 1))
        retrieval_results = [None] * len(queries)

        if retrieval_workers == 1:
            for q_idx, query in tqdm(enumerate(queries), desc="Retrieving", total=len(queries)):
                result_idx, query_solution, rerank_elapsed = retrieve_one(q_idx, query)
                retrieval_results[result_idx] = query_solution
                self.rerank_time += rerank_elapsed
        else:
            logger.info(f"Retrieving with {retrieval_workers} parallel workers.")
            with ThreadPoolExecutor(max_workers=retrieval_workers) as executor:
                futures = [
                    executor.submit(retrieve_one, q_idx, query)
                    for q_idx, query in enumerate(queries)
                ]
                for future in tqdm(as_completed(futures), desc="Retrieving", total=len(futures)):
                    result_idx, query_solution, rerank_elapsed = future.result()
                    retrieval_results[result_idx] = query_solution
                    self.rerank_time += rerank_elapsed

        retrieval_results = [result for result in retrieval_results if result is not None]

        retrieve_end_time = time.time()  # Record end time

        self.all_retrieval_time += retrieve_end_time - retrieve_start_time

        logger.info(f"Total Retrieval Time {self.all_retrieval_time:.2f}s")
        logger.info(f"Total Recognition Memory Time {self.rerank_time:.2f}s")
        logger.info(f"Total PPR Time {self.ppr_time:.2f}s")
        logger.info(f"Total Misc Time {self.all_retrieval_time - (self.rerank_time + self.ppr_time):.2f}s")

        # Evaluate retrieval
        if gold_docs is not None:
            k_list = [1, 2, 5, 10, 20, 30, 50, 100, 150, 200]
            overall_retrieval_result, example_retrieval_results = retrieval_recall_evaluator.calculate_metric_scores(gold_docs=gold_docs, retrieved_docs=[retrieval_result.docs for retrieval_result in retrieval_results], k_list=k_list)
            logger.info(f"Evaluation results for retrieval: {overall_retrieval_result}")

            return retrieval_results, overall_retrieval_result
        else:
            return retrieval_results

    def rag_qa(self,
               queries: List[str|QuerySolution],
               gold_docs: List[List[str]] = None,
               gold_answers: List[List[str]] = None) -> Tuple[List[QuerySolution], List[str], List[Dict]] | Tuple[List[QuerySolution], List[str], List[Dict], Dict, Dict]:
        """
        Performs retrieval-augmented generation enhanced QA using the SemFlowRAG 2 framework.

        This method can handle both string-based queries and pre-processed QuerySolution objects. Depending
        on its inputs, it returns answers only or additionally evaluate retrieval and answer quality using
        recall @ k, exact match and F1 score metrics.

        Parameters:
            queries (List[Union[str, QuerySolution]]): A list of queries, which can be either strings or
                QuerySolution instances. If they are strings, retrieval will be performed.
            gold_docs (Optional[List[List[str]]]): A list of lists containing gold-standard documents for
                each query. This is used if document-level evaluation is to be performed. Default is None.
            gold_answers (Optional[List[List[str]]]): A list of lists containing gold-standard answers for
                each query. Required if evaluation of question answering (QA) answers is enabled. Default
                is None.

        Returns:
            Union[
                Tuple[List[QuerySolution], List[str], List[Dict]],
                Tuple[List[QuerySolution], List[str], List[Dict], Dict, Dict]
            ]: A tuple that always includes:
                - List of QuerySolution objects containing answers and metadata for each query.
                - List of response messages for the provided queries.
                - List of metadata dictionaries for each query.
                If evaluation is enabled, the tuple also includes:
                - A dictionary with overall results from the retrieval phase (if applicable).
                - A dictionary with overall QA evaluation metrics (exact match and F1 scores).

        """
        if gold_answers is not None:
            qa_em_evaluator = QAExactMatch(global_config=self.global_config)
            qa_f1_evaluator = QAF1Score(global_config=self.global_config)

        # Retrieving (if necessary)
        overall_retrieval_result = None

        if not isinstance(queries[0], QuerySolution):
            if gold_docs is not None:
                queries, overall_retrieval_result = self.retrieve(queries=queries, gold_docs=gold_docs)
            else:
                queries = self.retrieve(queries=queries)

        # Performing QA
        queries_solutions, all_response_message, all_metadata = self.qa(queries)

        # Evaluating QA
        if gold_answers is not None:
            overall_qa_em_result, example_qa_em_results = qa_em_evaluator.calculate_metric_scores(
                gold_answers=gold_answers, predicted_answers=[qa_result.answer for qa_result in queries_solutions],
                aggregation_fn=np.max)
            overall_qa_f1_result, example_qa_f1_results = qa_f1_evaluator.calculate_metric_scores(
                gold_answers=gold_answers, predicted_answers=[qa_result.answer for qa_result in queries_solutions],
                aggregation_fn=np.max)

            # round off to 4 decimal places for QA results
            overall_qa_em_result.update(overall_qa_f1_result)
            overall_qa_results = overall_qa_em_result
            overall_qa_results = {k: round(float(v), 4) for k, v in overall_qa_results.items()}
            logger.info(f"Evaluation results for QA: {overall_qa_results}")

            # Save retrieval and QA results
            for idx, q in enumerate(queries_solutions):
                q.gold_answers = list(gold_answers[idx])
                if gold_docs is not None:
                    q.gold_docs = gold_docs[idx]

            return queries_solutions, all_response_message, all_metadata, overall_retrieval_result, overall_qa_results
        else:
            return queries_solutions, all_response_message, all_metadata

    def retrieve_dpr(self,
                     queries: List[str],
                     num_to_retrieve: int = None,
                     gold_docs: List[List[str]] = None) -> List[QuerySolution] | Tuple[List[QuerySolution], Dict]:
        """
        Performs retrieval using a DPR framework, which consists of several steps:
        - Dense passage scoring

        Parameters:
            queries: List[str]
                A list of query strings for which documents are to be retrieved.
            num_to_retrieve: int, optional
                The maximum number of documents to retrieve for each query. If not specified, defaults to
                the `retrieval_top_k` value defined in the global configuration.
            gold_docs: List[List[str]], optional
                A list of lists containing gold-standard documents corresponding to each query. Required
                if retrieval performance evaluation is enabled (`do_eval_retrieval` in global configuration).

        Returns:
            List[QuerySolution] or (List[QuerySolution], Dict)
                If retrieval performance evaluation is not enabled, returns a list of QuerySolution objects, each containing
                the retrieved documents and their scores for the corresponding query. If evaluation is enabled, also returns
                a dictionary containing the evaluation metrics computed over the retrieved results.

        Notes
        -----
        - Long queries with no relevant facts after reranking will default to results from dense passage retrieval.
        """
        retrieve_start_time = time.time()  # Record start time

        if num_to_retrieve is None:
            num_to_retrieve = self.global_config.retrieval_top_k

        if gold_docs is not None:
            retrieval_recall_evaluator = RetrievalRecall(global_config=self.global_config)

        if not self.ready_to_retrieve:
            self.prepare_retrieval_objects()

        self.get_query_embeddings(queries)

        retrieval_results = []

        for q_idx, query in tqdm(enumerate(queries), desc="Retrieving", total=len(queries)):
            logger.info('No facts found after reranking, return DPR results')
            sorted_doc_ids, sorted_doc_scores = self.dense_passage_retrieval(query)

            top_k_docs = [self.chunk_embedding_store.get_row(self.chunk_keys[idx])["content"] for idx in
                          sorted_doc_ids[:num_to_retrieve]]

            retrieval_results.append(
                QuerySolution(question=query, docs=top_k_docs, doc_scores=sorted_doc_scores[:num_to_retrieve]))

        retrieve_end_time = time.time()  # Record end time

        self.all_retrieval_time += retrieve_end_time - retrieve_start_time

        logger.info(f"Total Retrieval Time {self.all_retrieval_time:.2f}s")

        # Evaluate retrieval
        if gold_docs is not None:
            k_list = [1, 2, 5, 10, 20, 30, 50, 100, 150, 200]
            overall_retrieval_result, example_retrieval_results = retrieval_recall_evaluator.calculate_metric_scores(
                gold_docs=gold_docs, retrieved_docs=[retrieval_result.docs for retrieval_result in retrieval_results],
                k_list=k_list)
            logger.info(f"Evaluation results for retrieval: {overall_retrieval_result}")

            return retrieval_results, overall_retrieval_result
        else:
            return retrieval_results

    def rag_qa_dpr(self,
               queries: List[str|QuerySolution],
               gold_docs: List[List[str]] = None,
               gold_answers: List[List[str]] = None) -> Tuple[List[QuerySolution], List[str], List[Dict]] | Tuple[List[QuerySolution], List[str], List[Dict], Dict, Dict]:
        """
        Performs retrieval-augmented generation enhanced QA using a standard DPR framework.

        This method can handle both string-based queries and pre-processed QuerySolution objects. Depending
        on its inputs, it returns answers only or additionally evaluate retrieval and answer quality using
        recall @ k, exact match and F1 score metrics.

        Parameters:
            queries (List[Union[str, QuerySolution]]): A list of queries, which can be either strings or
                QuerySolution instances. If they are strings, retrieval will be performed.
            gold_docs (Optional[List[List[str]]]): A list of lists containing gold-standard documents for
                each query. This is used if document-level evaluation is to be performed. Default is None.
            gold_answers (Optional[List[List[str]]]): A list of lists containing gold-standard answers for
                each query. Required if evaluation of question answering (QA) answers is enabled. Default
                is None.

        Returns:
            Union[
                Tuple[List[QuerySolution], List[str], List[Dict]],
                Tuple[List[QuerySolution], List[str], List[Dict], Dict, Dict]
            ]: A tuple that always includes:
                - List of QuerySolution objects containing answers and metadata for each query.
                - List of response messages for the provided queries.
                - List of metadata dictionaries for each query.
                If evaluation is enabled, the tuple also includes:
                - A dictionary with overall results from the retrieval phase (if applicable).
                - A dictionary with overall QA evaluation metrics (exact match and F1 scores).

        """
        if gold_answers is not None:
            qa_em_evaluator = QAExactMatch(global_config=self.global_config)
            qa_f1_evaluator = QAF1Score(global_config=self.global_config)

        # Retrieving (if necessary)
        overall_retrieval_result = None

        if not isinstance(queries[0], QuerySolution):
            if gold_docs is not None:
                queries, overall_retrieval_result = self.retrieve_dpr(queries=queries, gold_docs=gold_docs)
            else:
                queries = self.retrieve_dpr(queries=queries)

        # Performing QA
        queries_solutions, all_response_message, all_metadata = self.qa(queries)

        # Evaluating QA
        if gold_answers is not None:
            overall_qa_em_result, example_qa_em_results = qa_em_evaluator.calculate_metric_scores(
                gold_answers=gold_answers, predicted_answers=[qa_result.answer for qa_result in queries_solutions],
                aggregation_fn=np.max)
            overall_qa_f1_result, example_qa_f1_results = qa_f1_evaluator.calculate_metric_scores(
                gold_answers=gold_answers, predicted_answers=[qa_result.answer for qa_result in queries_solutions],
                aggregation_fn=np.max)

            # round off to 4 decimal places for QA results
            overall_qa_em_result.update(overall_qa_f1_result)
            overall_qa_results = overall_qa_em_result
            overall_qa_results = {k: round(float(v), 4) for k, v in overall_qa_results.items()}
            logger.info(f"Evaluation results for QA: {overall_qa_results}")

            # Save retrieval and QA results
            for idx, q in enumerate(queries_solutions):
                q.gold_answers = list(gold_answers[idx])
                if gold_docs is not None:
                    q.gold_docs = gold_docs[idx]

            return queries_solutions, all_response_message, all_metadata, overall_retrieval_result, overall_qa_results
        else:
            return queries_solutions, all_response_message, all_metadata

    def qa(self, queries: List[QuerySolution]) -> Tuple[List[QuerySolution], List[str], List[Dict]]:
        """
        Executes question-answering (QA) inference using a provided set of query solutions and a language model.

        Parameters:
            queries: List[QuerySolution]
                A list of QuerySolution objects that contain the user queries, retrieved documents, and other related information.

        Returns:
            Tuple[List[QuerySolution], List[str], List[Dict]]
                A tuple containing:
                - A list of updated QuerySolution objects with the predicted answers embedded in them.
                - A list of raw response messages from the language model.
                - A list of metadata dictionaries associated with the results.
        """
        #Running inference for QA
        all_qa_messages = []

        def build_qa_messages(query_solution: QuerySolution, max_passage_chars: int | None = None):

            # obtain the retrieved docs
            retrieved_passages = query_solution.docs[:self.global_config.qa_top_k]

            prompt_user = ''
            for passage in retrieved_passages:
                if max_passage_chars is not None and len(passage) > max_passage_chars:
                    passage = passage[:max_passage_chars].rstrip()
                prompt_user += f'Wikipedia Title: {passage}\n\n'
            prompt_user += 'Question: ' + query_solution.question + '\nThought: '

            if self.prompt_template_manager.is_template_name_valid(name=f'rag_qa_{self.global_config.dataset}'):
                # find the corresponding prompt for this dataset
                prompt_dataset_name = self.global_config.dataset
            else:
                # the dataset does not have a customized prompt template yet
                logger.debug(
                    f"rag_qa_{self.global_config.dataset} does not have a customized prompt template. Using MUSIQUE's prompt template instead.")
                prompt_dataset_name = 'musique'
            return self.prompt_template_manager.render(name=f'rag_qa_{prompt_dataset_name}', prompt_user=prompt_user)

        for query_solution in tqdm(queries, desc="Collecting QA prompts"):
            all_qa_messages.append(build_qa_messages(query_solution))

        qa_workers = max(1, int(getattr(self.global_config, "qa_num_workers", 1) or 1))
        qa_temperature = getattr(self.global_config, "qa_temperature", None)
        qa_max_new_tokens = getattr(self.global_config, "qa_max_new_tokens", None)
        qa_infer_kwargs = {}
        if qa_temperature is not None:
            qa_infer_kwargs["temperature"] = qa_temperature
            logger.info(f"Running QA with temperature={qa_temperature}.")
        if qa_max_new_tokens is not None and qa_max_new_tokens > 0:
            qa_infer_kwargs["max_completion_tokens"] = int(qa_max_new_tokens)
            logger.info(f"Running QA with max_new_tokens={qa_max_new_tokens}.")

        def is_context_length_error(exc: Exception) -> bool:
            candidates = []
            seen = set()

            def collect(candidate):
                if candidate is None or id(candidate) in seen:
                    return
                seen.add(id(candidate))
                candidates.append(candidate)

                last_attempt = getattr(candidate, "last_attempt", None)
                if last_attempt is not None:
                    try:
                        collect(last_attempt.exception())
                    except Exception:
                        pass

                collect(getattr(candidate, "__cause__", None))
                collect(getattr(candidate, "__context__", None))

            collect(exc)
            for candidate in candidates:
                message = str(candidate).lower()
                if (
                    "maximum context length" in message
                    or "context length" in message
                    or "reduce the length of the input prompt" in message
                    or "input tokens" in message
                ):
                    return True
            return False

        def infer_qa_with_context_retry(query_idx: int, qa_messages):
            try:
                return self.llm_model.infer(qa_messages, **qa_infer_kwargs)
            except Exception as exc:
                if not is_context_length_error(exc):
                    raise

                logger.warning(
                    "QA prompt exceeded context window at query_idx=%s; retrying with truncated passages.",
                    query_idx,
                )
                last_exc = exc
                for max_passage_chars in (2400, 1800, 1200, 800, 500):
                    retry_messages = build_qa_messages(queries[query_idx], max_passage_chars=max_passage_chars)
                    try:
                        return self.llm_model.infer(retry_messages, **qa_infer_kwargs)
                    except Exception as retry_exc:
                        if not is_context_length_error(retry_exc):
                            raise
                        last_exc = retry_exc
                raise last_exc

        if qa_workers == 1:
            all_qa_results = [
                infer_qa_with_context_retry(query_idx, qa_messages)
                for query_idx, qa_messages in tqdm(enumerate(all_qa_messages), desc="QA Reading", total=len(all_qa_messages))
            ]
        else:
            logger.info(f"Running QA with {qa_workers} parallel workers.")
            all_qa_results = [None] * len(all_qa_messages)
            with ThreadPoolExecutor(max_workers=qa_workers) as executor:
                futures = [
                    executor.submit(infer_qa_with_context_retry, query_idx, qa_messages)
                    for query_idx, qa_messages in enumerate(all_qa_messages)
                ]
                for future_idx, future in tqdm(
                    enumerate(futures),
                    desc="QA Reading",
                    total=len(futures),
                ):
                    all_qa_results[future_idx] = future.result()

        all_response_message, all_metadata, all_cache_hit = zip(*all_qa_results)
        all_response_message, all_metadata = list(all_response_message), list(all_metadata)

        #Process responses and extract predicted answers.
        queries_solutions = []
        for query_solution_idx, query_solution in tqdm(enumerate(queries), desc="Extraction Answers from LLM Response"):
            response_content = all_response_message[query_solution_idx]
            try:
                pred_ans = response_content.split('Answer:')[1].strip()
            except Exception as e:
                logger.warning(f"Error in parsing the answer from the raw LLM QA inference response: {str(e)}!")
                pred_ans = response_content

            query_solution.answer = pred_ans
            queries_solutions.append(query_solution)

        return queries_solutions, all_response_message, all_metadata

    def add_fact_edges(self, chunk_ids: List[str], chunk_triples: List[Tuple]):
        """
        Adds fact edges from given triples to the graph.

        The method processes chunks of triples, computes unique identifiers
        for entities and relations, and updates various internal statistics
        to build and maintain the graph structure. Entities are uniquely
        identified and linked based on their relationships.

        Parameters:
            chunk_ids: List[str]
                A list of unique identifiers for the chunks being processed.
            chunk_triples: List[Tuple]
                A list of tuples representing triples to process. Each triple
                consists of a subject, predicate, and object.

        Raises:
            Does not explicitly raise exceptions within the provided function logic.
        """

        if "name" in self.graph.vs:
            current_graph_nodes = set(self.graph.vs["name"])
        else:
            current_graph_nodes = set()

        logger.info(f"Adding OpenIE triples to graph.")

        for chunk_key, triples in tqdm(zip(chunk_ids, chunk_triples)):
            entities_in_chunk = set()

            for triple in triples:
                triple = tuple(triple)

                node_key = compute_mdhash_id(content=triple[0], prefix=("entity-"))
                node_2_key = compute_mdhash_id(content=triple[2], prefix=("entity-"))

                if chunk_key not in current_graph_nodes:
                    self.node_to_node_stats[(node_key, node_2_key)] = self.node_to_node_stats.get(
                        (node_key, node_2_key), 0.0) + 1
                    self.node_to_node_stats[(node_2_key, node_key)] = self.node_to_node_stats.get(
                        (node_2_key, node_key), 0.0) + 1

                entities_in_chunk.add(node_key)
                entities_in_chunk.add(node_2_key)

            for node in entities_in_chunk:
                self.ent_node_to_chunk_ids[node] = self.ent_node_to_chunk_ids.get(node, set()).union(set([chunk_key]))

    def add_passage_edges(self, chunk_ids: List[str], chunk_triple_entities: List[List[str]]):
        """
        Connect passage nodes with their extracted entity nodes.

        In directed graphs the connection is bidirectional so passages can both
        collect probability from entities and provide graph connectivity back to
        co-mentioned entities. These edges are structural only; query/diversity
        weighting is deliberately skipped in directed PPR.
        """

        logger.info(f"Connecting passage nodes to phrase nodes.")

        add_reverse = self.graph.is_directed()

        for idx, chunk_key in tqdm(enumerate(chunk_ids)):
            for chunk_ent in chunk_triple_entities[idx]:
                node_key = compute_mdhash_id(chunk_ent, prefix="entity-")
                self.node_to_node_stats[(node_key, chunk_key)] = 1.0
                if add_reverse:
                    self.node_to_node_stats[(chunk_key, node_key)] = 1.0

    def add_synonymy_edges(self):
        """
        Adds synonymy edges between similar nodes in the graph to enhance connectivity by identifying and linking synonym entities.

        This method performs key operations to compute and add synonymy edges. It first retrieves embeddings for all nodes, then conducts
        a nearest neighbor (KNN) search to find similar nodes. These similar nodes are identified based on a score threshold, and edges
        are added to represent the synonym relationship.

        Attributes:
            entity_id_to_row: dict (populated within the function). Maps each entity ID to its corresponding row data, where rows
                              contain `content` of entities used for comparison.
            entity_embedding_store: Manages retrieval of texts and embeddings for all rows related to entities.
            global_config: Configuration object that defines parameters such as `synonymy_edge_topk`, `synonymy_edge_sim_threshold`,
                           `synonymy_edge_query_batch_size`, and `synonymy_edge_key_batch_size`.
            node_to_node_stats: dict. Stores scores for edges between nodes representing their relationship.

        """
        logger.info(f"Expanding graph with synonymy edges")

        self.entity_id_to_row = self.entity_embedding_store.get_all_id_to_rows()
        entity_node_keys = list(self.entity_id_to_row.keys())

        logger.info(f"Performing KNN retrieval for each phrase nodes ({len(entity_node_keys)}).")

        entity_embs = self.entity_embedding_store.get_embeddings(entity_node_keys)

        # Here we build synonymy edges only between newly inserted phrase nodes and all phrase nodes in the storage to reduce cost for incremental graph updates
        query_node_key2knn_node_keys = retrieve_knn(query_ids=entity_node_keys,
                                                    key_ids=entity_node_keys,
                                                    query_vecs=entity_embs,
                                                    key_vecs=entity_embs,
                                                    k=self.global_config.synonymy_edge_topk,
                                                    query_batch_size=self.global_config.synonymy_edge_query_batch_size,
                                                    key_batch_size=self.global_config.synonymy_edge_key_batch_size)

        num_synonym_triple = 0
        synonym_candidates = []  # [(node key, [(synonym node key, corresponding score), ...]), ...]

        for node_key in tqdm(query_node_key2knn_node_keys.keys(), total=len(query_node_key2knn_node_keys)):
            synonyms = []

            entity = self.entity_id_to_row[node_key]["content"]

            if len(re.sub('[^A-Za-z0-9]', '', entity)) > 2:
                nns = query_node_key2knn_node_keys[node_key]

                num_nns = 0
                for nn, score in zip(nns[0], nns[1]):
                    if score < self.global_config.synonymy_edge_sim_threshold or num_nns > 100:
                        break

                    nn_phrase = self.entity_id_to_row[nn]["content"]

                    if nn != node_key and nn_phrase != '':
                        sim_edge = (node_key, nn)
                        synonyms.append((nn, score))
                        num_synonym_triple += 1

                        self.node_to_node_stats[sim_edge] = score  # Need to seriously discuss on this
                        num_nns += 1

            synonym_candidates.append((node_key, synonyms))

    def load_existing_openie(self, chunk_keys: List[str]) -> Tuple[List[dict], Set[str]]:
        """
        Loads existing OpenIE results from the specified file if it exists and combines
        them with new content while standardizing indices. If the file does not exist or
        is configured to be re-initialized from scratch with the flag `force_openie_from_scratch`,
        it prepares new entries for processing.

        Args:
            chunk_keys (List[str]): A list of chunk keys that represent identifiers
                                     for the content to be processed.

        Returns:
            Tuple[List[dict], Set[str]]: A tuple where the first element is the existing OpenIE
                                         information (if any) loaded from the file, and the
                                         second element is a set of chunk keys that still need to
                                         be saved or processed.
        """

        # combine openie_results with contents already in file, if file exists
        chunk_keys_to_save = set()

        if not self.global_config.force_openie_from_scratch and os.path.isfile(self.openie_results_path):
            openie_results = json.load(open(self.openie_results_path))
            all_openie_info = openie_results.get('docs', [])

            #Standardizing indices for OpenIE Files.

            renamed_openie_info = []
            for openie_info in all_openie_info:
                openie_info['idx'] = compute_mdhash_id(openie_info['passage'], 'chunk-')
                renamed_openie_info.append(openie_info)

            all_openie_info = renamed_openie_info

            existing_openie_keys = set([info['idx'] for info in all_openie_info])

            for chunk_key in chunk_keys:
                if chunk_key not in existing_openie_keys:
                    chunk_keys_to_save.add(chunk_key)
        else:
            all_openie_info = []
            chunk_keys_to_save = chunk_keys

        return all_openie_info, chunk_keys_to_save

    def merge_openie_results(self,
                             all_openie_info: List[dict],
                             chunks_to_save: Dict[str, dict],
                             ner_results_dict: Dict[str, NerRawOutput],
                             triple_results_dict: Dict[str, TripleRawOutput]) -> List[dict]:
        """
        Merges OpenIE extraction results with corresponding passage and metadata.

        This function integrates the OpenIE extraction results, including named-entity
        recognition (NER) entities and triples, with their respective text passages
        using the provided chunk keys. The resulting merged data is appended to
        the `all_openie_info` list containing dictionaries with combined and organized
        data for further processing or storage.

        Parameters:
            all_openie_info (List[dict]): A list to hold dictionaries of merged OpenIE
                results and metadata for all chunks.
            chunks_to_save (Dict[str, dict]): A dict of chunk identifiers (keys) to process
                and merge OpenIE results to dictionaries with `hash_id` and `content` keys.
            ner_results_dict (Dict[str, NerRawOutput]): A dictionary mapping chunk keys
                to their corresponding NER extraction results.
            triple_results_dict (Dict[str, TripleRawOutput]): A dictionary mapping chunk
                keys to their corresponding OpenIE triple extraction results.

        Returns:
            List[dict]: The `all_openie_info` list containing dictionaries with merged
            OpenIE results, metadata, and the passage content for each chunk.

        """

        for chunk_key, row in chunks_to_save.items():
            passage = row['content']
            try:
                chunk_openie_info = {'idx': chunk_key, 'passage': passage,
                                 'extracted_entities': ner_results_dict[chunk_key].unique_entities,
                                 'extracted_triples': triple_results_dict[chunk_key].triples}
            except Exception as e:
                logger.error(f"Error processing chunk {chunk_key}: {e}")
                chunk_openie_info = {'idx': chunk_key, 'passage': passage,
                                 'extracted_entities': [],
                                 'extracted_triples': []}
            all_openie_info.append(chunk_openie_info)

        return all_openie_info

    def save_openie_results(self, all_openie_info: List[dict]):
        """
        Computes statistics on extracted entities from OpenIE results and saves the aggregated data in a
        JSON file. The function calculates the average character and word lengths of the extracted entities
        and writes them along with the provided OpenIE information to a file.

        Parameters:
            all_openie_info : List[dict]
                List of dictionaries, where each dictionary represents information from OpenIE, including
                extracted entities.
        """

        sum_phrase_chars = sum([len(e) for chunk in all_openie_info for e in chunk['extracted_entities']])
        sum_phrase_words = sum([len(e.split()) for chunk in all_openie_info for e in chunk['extracted_entities']])
        num_phrases = sum([len(chunk['extracted_entities']) for chunk in all_openie_info])

        if len(all_openie_info) > 0:
            # Avoid division by zero if there are no phrases
            if num_phrases > 0:
                avg_ent_chars = round(sum_phrase_chars / num_phrases, 4)
                avg_ent_words = round(sum_phrase_words / num_phrases, 4)
            else:
                avg_ent_chars = 0
                avg_ent_words = 0
                
            openie_dict = {
                'docs': all_openie_info,
                'avg_ent_chars': avg_ent_chars,
                'avg_ent_words': avg_ent_words
            }
            
            with open(self.openie_results_path, 'w') as f:
                json.dump(openie_dict, f)
            logger.info(f"OpenIE results saved to {self.openie_results_path}")

    def augment_graph(self):
        """
        Provides utility functions to augment a graph by adding new nodes and edges.
        It ensures that the graph structure is extended to include additional components,
        and logs the completion status along with printing the updated graph information.
        """

        self.add_new_nodes()
        self.add_new_edges()

        logger.info(f"Graph construction completed!")
        print(self.get_graph_info())

    def add_new_nodes(self):
        """
        Adds new entity and passage nodes to the graph from the embedding stores.

        This method identifies and adds new nodes to the graph by comparing existing nodes
        in the graph and nodes retrieved from the entity and passage embedding stores.
        The method checks attributes and ensures no duplicates are added. New nodes are
        prepared and added in bulk to optimize graph updates.
        """

        existing_nodes = {v["name"]: v for v in self.graph.vs if "name" in v.attributes()}

        entity_to_rows = self.entity_embedding_store.get_all_id_to_rows()
        passage_to_rows = self.chunk_embedding_store.get_all_id_to_rows()

        node_to_rows = entity_to_rows
        node_to_rows.update(passage_to_rows)

        new_nodes = {}
        for node_id, node in node_to_rows.items():
            node['name'] = node_id
            if node_id not in existing_nodes:
                for k, v in node.items():
                    if k not in new_nodes:
                        new_nodes[k] = []
                    new_nodes[k].append(v)

        if len(new_nodes) > 0:
            self.graph.add_vertices(n=len(next(iter(new_nodes.values()))), attributes=new_nodes)

    def remove_chunk_vertices_from_graph(self) -> int:
        """
        Removes legacy chunk vertices from the graph.

        Chunks are still stored as retrievable documents, but they are no longer graph
        vertices. This keeps old graph pickles compatible with the entity-only graph.
        """
        if "name" not in self.graph.vs.attribute_names():
            return 0

        chunk_ids = set(self.chunk_embedding_store.get_all_ids())
        vertex_ids_to_delete = [
            idx for idx, node_name in enumerate(self.graph.vs["name"])
            if node_name in chunk_ids
        ]

        if len(vertex_ids_to_delete) > 0:
            self.graph.delete_vertices(vertex_ids_to_delete)
            logger.info(f"Removed {len(vertex_ids_to_delete)} legacy chunk vertices from graph.")

        return len(vertex_ids_to_delete)

    def add_new_edges(self) -> int:
        """
        Processes edges from `node_to_node_stats` to add them into a graph object while
        managing adjacency lists, validating edges, and logging invalid edge cases.
        """

        graph_adj_list = defaultdict(dict)
        graph_inverse_adj_list = defaultdict(dict)
        edge_source_node_keys = []
        edge_target_node_keys = []
        edge_metadata = []
        for edge, weight in self.node_to_node_stats.items():
            if edge[0] == edge[1]: continue
            graph_adj_list[edge[0]][edge[1]] = weight
            graph_inverse_adj_list[edge[1]][edge[0]] = weight

            edge_source_node_keys.append(edge[0])
            edge_target_node_keys.append(edge[1])
            edge_metadata.append({
                "weight": weight
            })

        valid_edges, valid_weights = [], {"weight": []}
        current_node_ids = set(self.graph.vs["name"])
        vertex_names = self.graph.vs["name"] if self.graph.vcount() > 0 else []
        graph_is_directed = self.graph.is_directed()
        existing_edges = set()
        for src_idx, dst_idx in self.graph.get_edgelist():
            src_name = vertex_names[src_idx]
            dst_name = vertex_names[dst_idx]
            edge_key = (src_name, dst_name) if graph_is_directed else tuple(sorted((src_name, dst_name)))
            existing_edges.add(edge_key)

        for source_node_id, target_node_id, edge_d in zip(edge_source_node_keys, edge_target_node_keys, edge_metadata):
            if source_node_id in current_node_ids and target_node_id in current_node_ids:
                edge_key = (
                    (source_node_id, target_node_id)
                    if graph_is_directed
                    else tuple(sorted((source_node_id, target_node_id)))
                )
                if edge_key in existing_edges:
                    continue
                valid_edges.append((source_node_id, target_node_id))
                weight = edge_d.get("weight", 1.0)
                valid_weights["weight"].append(weight)
                existing_edges.add(edge_key)
            else:
                logger.warning(f"Edge {source_node_id} -> {target_node_id} is not valid.")
        if len(valid_edges) > 0:
            self.graph.add_edges(
                valid_edges,
                attributes=valid_weights
            )
        return len(valid_edges)

    def save_igraph(self):
        logger.info(
            f"Writing graph with {len(self.graph.vs())} nodes, {len(self.graph.es())} edges"
        )
        self.graph.write_pickle(self._graph_pickle_filename)
        logger.info(f"Saving graph completed!")

    def get_graph_info(self) -> Dict:
        """
        Obtains detailed information about the entity graph.

        This method calculates various statistics about the graph based on the
        stores and node-to-node relationships, including counts of phrase nodes,
        extracted triples, synonymy triples, and total triples.

        Returns:
            Dict
                A dictionary containing the following keys and their respective values:
                - num_phrase_nodes: The number of unique phrase nodes.
                - num_total_nodes: The total number of graph nodes.
                - num_extracted_triples: The number of unique extracted triples.
                - num_synonymy_triples: The number of synonymy triples.
                - num_total_triples: The total number of triples.
        """
        graph_info = {}

        # get # of phrase nodes
        phrase_nodes_keys = self.entity_embedding_store.get_all_ids()
        graph_info["num_phrase_nodes"] = len(set(phrase_nodes_keys))

        # get # of total nodes
        graph_info["num_total_nodes"] = self.graph.vcount()

        # get # of extracted triples
        graph_info["num_extracted_triples"] = len(self.fact_embedding_store.get_all_ids())

        graph_info['num_synonymy_triples'] = len(self.node_to_node_stats) - graph_info["num_extracted_triples"]

        # get # of total triples
        graph_info["num_total_triples"] = len(self.node_to_node_stats)

        return graph_info

    def prepare_retrieval_objects(self):
        """
        Prepares various in-memory objects and attributes necessary for fast retrieval processes, such as embedding data and graph relationships, ensuring consistency
        and alignment with the underlying graph structure.
        """

        logger.info("Preparing for fast retrieval.")

        logger.info("Loading keys.")
        self.query_to_embedding: Dict = {'triple': {}, 'passage': {}}

        self.entity_node_keys: List = list(self.entity_embedding_store.get_all_ids()) # a list of phrase node keys
        self.chunk_keys: List = list(self.chunk_embedding_store.get_all_ids())
        self.passage_node_keys: List = self.chunk_keys
        self.fact_node_keys: List = list(self.fact_embedding_store.get_all_ids())
        graph_was_modified = False

        # Check if the graph has the expected number of nodes
        expected_node_count = len(self.entity_node_keys) + len(self.passage_node_keys)
        actual_node_count = self.graph.vcount()
        
        if expected_node_count != actual_node_count:
            logger.warning(f"Graph node count mismatch: expected {expected_node_count}, got {actual_node_count}")
            pre_nodes = self.graph.vcount()
            self.add_new_nodes()
            graph_was_modified = graph_was_modified or self.graph.vcount() != pre_nodes

        # Create mapping from node name to vertex index
        try:
            igraph_name_to_idx = {node["name"]: idx for idx, node in enumerate(self.graph.vs)} # from node key to the index in the backbone graph
            self.node_name_to_vertex_idx = igraph_name_to_idx
            
            # Check if all entity and passage nodes are in the graph
            missing_entity_nodes = [node_key for node_key in self.entity_node_keys if node_key not in igraph_name_to_idx]
            missing_passage_nodes = [node_key for node_key in self.passage_node_keys if node_key not in igraph_name_to_idx]
            
            if missing_entity_nodes or missing_passage_nodes:
                logger.warning(f"Missing nodes in graph: {len(missing_entity_nodes)} entity nodes, {len(missing_passage_nodes)} passage nodes")
                # If nodes are missing, rebuild the graph
                pre_nodes = self.graph.vcount()
                self.add_new_nodes()
                graph_was_modified = graph_was_modified or self.graph.vcount() != pre_nodes
                # Update the mapping
                igraph_name_to_idx = {node["name"]: idx for idx, node in enumerate(self.graph.vs)}
                self.node_name_to_vertex_idx = igraph_name_to_idx
            
            self.entity_node_idxs = [igraph_name_to_idx[node_key] for node_key in self.entity_node_keys] # a list of backbone graph node index
            self.passage_node_idxs = [igraph_name_to_idx[node_key] for node_key in self.passage_node_keys] # a list of backbone passage node index
        except Exception as e:
            logger.error(f"Error creating node index mapping: {str(e)}")
            # Initialize with empty lists if mapping fails
            self.node_name_to_vertex_idx = {}
            self.entity_node_idxs = []
            self.passage_node_idxs = []

        logger.info("Loading embeddings.")
        self.entity_embeddings = np.array(self.entity_embedding_store.get_embeddings(self.entity_node_keys))
        self.passage_embeddings = np.array(self.chunk_embedding_store.get_embeddings(self.chunk_keys))

        self.fact_embeddings = np.array(self.fact_embedding_store.get_embeddings(self.fact_node_keys))

        all_openie_info, chunk_keys_to_process = self.load_existing_openie([])

        self.proc_triples_to_docs = {}

        for doc in all_openie_info:
            triples = flatten_facts([doc['extracted_triples']])
            for triple in triples:
                if len(triple) == 3:
                    proc_triple = tuple(text_processing(list(triple)))
                    self.proc_triples_to_docs[str(proc_triple)] = self.proc_triples_to_docs.get(str(proc_triple), set()).union(set([doc['idx']]))

        if self.ent_node_to_chunk_ids is None:
            ner_results_dict, triple_results_dict = reformat_openie_results(all_openie_info)

            # Check if the lengths match
            if not (len(self.chunk_keys) == len(ner_results_dict) == len(triple_results_dict)):
                logger.warning(f"Length mismatch: chunk_keys={len(self.chunk_keys)}, ner_results_dict={len(ner_results_dict)}, triple_results_dict={len(triple_results_dict)}")
                
                # If there are missing keys, create empty entries for them
                for chunk_id in self.chunk_keys:
                    if chunk_id not in ner_results_dict:
                        ner_results_dict[chunk_id] = NerRawOutput(
                            chunk_id=chunk_id,
                            response=None,
                            metadata={},
                            unique_entities=[]
                        )
                    if chunk_id not in triple_results_dict:
                        triple_results_dict[chunk_id] = TripleRawOutput(
                            chunk_id=chunk_id,
                            response=None,
                            metadata={},
                            triples=[]
                        )

            # prepare data_store
            chunk_triples = [[text_processing(t) for t in triple_results_dict[chunk_id].triples] for chunk_id in self.chunk_keys]
            _, chunk_triple_entities = extract_entity_nodes(chunk_triples)

            self.node_to_node_stats = {}
            self.ent_node_to_chunk_ids = {}
            self.add_fact_edges(self.chunk_keys, chunk_triples)
            self.add_passage_edges(self.chunk_keys, chunk_triple_entities)

        if hasattr(self, 'node_to_node_stats') and len(self.node_to_node_stats) > 0:
            added_edges = self.add_new_edges()
            if added_edges > 0:
                logger.info(f"Added {added_edges} missing graph edges for retrieval.")
                graph_was_modified = True

        # Directed-PPR assets: ensure self-loop placeholders exist and the prior_norm /
        # diversity_norm caches are populated. Supports loading older graph pickles that
        # predate the innovation without forcing a full re-index.
        if self.global_config.enable_directed_ppr:
            pre_edges = self.graph.ecount()
            self._ensure_self_loops()
            if self.graph.ecount() != pre_edges:
                graph_was_modified = True

            edge_attrs = self.graph.es.attributes() if self.graph.ecount() > 0 else []
            prior_values = self.graph.es['prior_norm'] if 'prior_norm' in edge_attrs else []
            prior_has_missing = any(v is None for v in prior_values)
            if graph_was_modified or 'prior_norm' not in edge_attrs or prior_has_missing:
                self.precompute_prior_norm()
                graph_was_modified = True

            vs_attrs = self.graph.vs.attributes() if self.graph.vcount() > 0 else []
            diversity_values = self.graph.vs['diversity_norm'] if 'diversity_norm' in vs_attrs else []
            diversity_has_missing = any(v is None for v in diversity_values)
            if graph_was_modified or 'diversity_norm' not in vs_attrs or diversity_has_missing:
                self.precompute_entity_diversity()
                graph_was_modified = True
            else:
                self.entity_diversity_norm = np.asarray(
                    self.graph.vs['diversity_norm'], dtype=np.float32
                )

        if graph_was_modified:
            self.save_igraph()

        self._precompute_retrieval_caches()

        self.ready_to_retrieve = True

    def _precompute_retrieval_caches(self):
        """One-shot precompute of query-independent retrieval structures.

        Replaces per-query Python loops over N_nodes / N_edges / N_chunks inside
        `graph_search_with_fact_entities` and `build_directed_edge_weights` with
        numpy gathers + segment reductions. Called once at the end of
        `prepare_retrieval_objects`; the graph is treated as frozen afterwards.
        """
        n_nodes = self.graph.vcount()
        n_edges = self.graph.ecount()

        vertex_names = list(self.graph.vs['name']) if n_nodes > 0 else []
        self._cached_vertex_names = vertex_names

        chunk_keys_set = set(self.chunk_keys)
        node_is_passage = np.fromiter(
            (name in chunk_keys_set for name in vertex_names),
            dtype=bool,
            count=n_nodes,
        )
        self._cached_node_is_passage = node_is_passage
        self._cached_node_is_entity = ~node_is_passage

        chunk_to_passage_v_idx = np.full(len(self.chunk_keys), -1, dtype=np.int64)
        for i, chunk_key in enumerate(self.chunk_keys):
            v = self.node_name_to_vertex_idx.get(chunk_key, None)
            if v is not None:
                chunk_to_passage_v_idx[i] = int(v)
        self._cached_chunk_to_passage_v_idx = chunk_to_passage_v_idx

        if n_edges > 0:
            edgelist = np.asarray(self.graph.get_edgelist(), dtype=np.int64)
            u_idx = edgelist[:, 0]
            v_idx = edgelist[:, 1]
        else:
            u_idx = np.zeros(0, dtype=np.int64)
            v_idx = np.zeros(0, dtype=np.int64)
        is_self_loop = u_idx == v_idx
        src_is_passage = node_is_passage[u_idx] if n_edges > 0 else np.zeros(0, dtype=bool)
        dst_is_passage = node_is_passage[v_idx] if n_edges > 0 else np.zeros(0, dtype=bool)
        self._cached_edgelist_u = u_idx
        self._cached_edgelist_v = v_idx
        self._cached_is_self_loop = is_self_loop
        self._cached_is_passage_edge = (src_is_passage | dst_is_passage) & ~is_self_loop
        self._cached_is_entity_edge = (~src_is_passage & ~dst_is_passage) & ~is_self_loop

        edge_attrs = self.graph.es.attributes() if n_edges > 0 else []
        if 'prior_norm' in edge_attrs:
            self._cached_prior_norm = np.asarray(
                [0.0 if value is None else value for value in self.graph.es['prior_norm']],
                dtype=np.float32,
            )
        else:
            self._cached_prior_norm = np.zeros(n_edges, dtype=np.float32)

        vs_attrs = self.graph.vs.attributes() if n_nodes > 0 else []
        if 'diversity_norm' in vs_attrs:
            self._cached_div_norm = np.asarray(
                [0.5 if value is None else value for value in self.graph.vs['diversity_norm']],
                dtype=np.float32,
            )
        else:
            self._cached_div_norm = np.full(n_nodes, 0.5, dtype=np.float32)

        chunk_key_to_idx = {ck: idx for idx, ck in enumerate(self.chunk_keys)}
        entity_v_idx_list: List[int] = []
        chunk_indices_chunks: List[np.ndarray] = []
        ent_map = self.ent_node_to_chunk_ids or {}
        for v_id, ent_key in enumerate(vertex_names):
            if not self._cached_node_is_entity[v_id]:
                continue
            ck_set = ent_map.get(ent_key, None)
            if not ck_set:
                continue
            idxs = [chunk_key_to_idx[ck] for ck in ck_set if ck in chunk_key_to_idx]
            if not idxs:
                continue
            entity_v_idx_list.append(v_id)
            chunk_indices_chunks.append(np.asarray(idxs, dtype=np.int64))

        if entity_v_idx_list:
            sizes = np.fromiter((arr.size for arr in chunk_indices_chunks), dtype=np.int64,
                                count=len(chunk_indices_chunks))
            indptr = np.empty(len(sizes) + 1, dtype=np.int64)
            indptr[0] = 0
            np.cumsum(sizes, out=indptr[1:])
            self._entity_chunk_indices = np.concatenate(chunk_indices_chunks)
            self._entity_chunk_indptr = indptr
            self._entity_v_idx_with_chunks = np.asarray(entity_v_idx_list, dtype=np.int64)
            assert np.all(np.diff(indptr) > 0), "entity CSR indptr must be strictly increasing"
        else:
            self._entity_chunk_indices = np.zeros(0, dtype=np.int64)
            self._entity_chunk_indptr = np.zeros(1, dtype=np.int64)
            self._entity_v_idx_with_chunks = np.zeros(0, dtype=np.int64)

        logger.info(
            "Precomputed retrieval caches: %d nodes, %d edges, %d entities-with-chunks (%d chunk refs).",
            n_nodes, n_edges,
            int(self._entity_v_idx_with_chunks.size),
            int(self._entity_chunk_indices.size),
        )

        # Entity-only subgraph for the Innovation-vs-Vanilla comparison experiment.
        # Built once at retrieval prep; igraph.subgraph() preserves vertex / edge attributes
        # (incl. 'weight', 'prior_norm', 'diversity_norm'). Used only as a read-only side-channel
        # by _run_entity_only_ppr_experiment; main retrieval path is unaffected.
        if n_nodes > 0 and self._cached_node_is_entity.any():
            entity_v_indices = np.flatnonzero(self._cached_node_is_entity).astype(np.int64)
            self._eo_full_v_indices = entity_v_indices
            self._eo_graph = self.graph.subgraph(entity_v_indices.tolist())
            logger.info(
                "Entity-only subgraph: %d vertices, %d edges (vs full %d/%d).",
                self._eo_graph.vcount(), self._eo_graph.ecount(), n_nodes, n_edges,
            )
        else:
            self._eo_full_v_indices = np.zeros(0, dtype=np.int64)
            self._eo_graph = None

    def get_query_embeddings(self, queries: List[str] | List[QuerySolution]):
        """
        Retrieves embeddings for given queries and updates the internal query-to-embedding mapping. The method determines whether each query
        is already present in the `self.query_to_embedding` dictionary under the keys 'triple' and 'passage'. If a query is not present in
        either, it is encoded into embeddings using the embedding model and stored.

        Args:
            queries List[str] | List[QuerySolution]: A list of query strings or QuerySolution objects. Each query is checked for
            its presence in the query-to-embedding mappings.
        """

        all_query_strings = []
        for query in queries:
            if isinstance(query, QuerySolution) and (
                    query.question not in self.query_to_embedding['triple'] or query.question not in
                    self.query_to_embedding['passage']):
                all_query_strings.append(query.question)
            elif query not in self.query_to_embedding['triple'] or query not in self.query_to_embedding['passage']:
                all_query_strings.append(query)

        if len(all_query_strings) > 0:
            # get all query embeddings
            logger.info(f"Encoding {len(all_query_strings)} queries for query_to_fact.")
            query_embeddings_for_triple = self.embedding_model.batch_encode(all_query_strings,
                                                                            instruction=get_query_instruction('query_to_fact'),
                                                                            norm=True)
            for query, embedding in zip(all_query_strings, query_embeddings_for_triple):
                self.query_to_embedding['triple'][query] = embedding

            logger.info(f"Encoding {len(all_query_strings)} queries for query_to_passage.")
            query_embeddings_for_passage = self.embedding_model.batch_encode(all_query_strings,
                                                                             instruction=get_query_instruction('query_to_passage'),
                                                                             norm=True)
            for query, embedding in zip(all_query_strings, query_embeddings_for_passage):
                self.query_to_embedding['passage'][query] = embedding

    def get_fact_scores(self, query: str) -> np.ndarray:
        """
        Retrieves and computes normalized similarity scores between the given query and pre-stored fact embeddings.

        Parameters:
        query : str
            The input query text for which similarity scores with fact embeddings
            need to be computed.

        Returns:
        numpy.ndarray
            A normalized array of similarity scores between the query and fact
            embeddings. The shape of the array is determined by the number of
            facts.

        Raises:
        KeyError
            If no embedding is found for the provided query in the stored query
            embeddings dictionary.
        """
        query_embedding = self.query_to_embedding['triple'].get(query, None)
        if query_embedding is None:
            query_embedding = self.embedding_model.batch_encode(query,
                                                                instruction=get_query_instruction('query_to_fact'),
                                                                norm=True)

        # Check if there are any facts
        if len(self.fact_embeddings) == 0:
            logger.warning("No facts available for scoring. Returning empty array.")
            return np.array([])
            
        try:
            query_fact_scores = np.dot(self.fact_embeddings, query_embedding.T) # shape: (#facts, )
            query_fact_scores = np.squeeze(query_fact_scores) if query_fact_scores.ndim == 2 else query_fact_scores
            query_fact_scores = min_max_normalize(query_fact_scores)
            return query_fact_scores
        except Exception as e:
            logger.error(f"Error computing fact scores: {str(e)}")
            return np.array([])

    def dense_passage_retrieval(self, query: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Conduct dense passage retrieval to find relevant documents for a query.

        This function processes a given query using a pre-trained embedding model
        to generate query embeddings. The similarity scores between the query
        embedding and passage embeddings are computed using dot product, followed
        by score normalization. Finally, the function ranks the documents based
        on their similarity scores and returns the ranked document identifiers
        and their scores.

        Parameters
        ----------
        query : str
            The input query for which relevant passages should be retrieved.

        Returns
        -------
        tuple : Tuple[np.ndarray, np.ndarray]
            A tuple containing two elements:
            - A list of sorted document identifiers based on their relevance scores.
            - A numpy array of the normalized similarity scores for the corresponding
              documents.
        """
        query_embedding = self.query_to_embedding['passage'].get(query, None)
        if query_embedding is None:
            query_embedding = self.embedding_model.batch_encode(query,
                                                                instruction=get_query_instruction('query_to_passage'),
                                                                norm=True)
        query_doc_scores = np.dot(self.passage_embeddings, query_embedding.T)
        query_doc_scores = np.squeeze(query_doc_scores) if query_doc_scores.ndim == 2 else query_doc_scores
        query_doc_scores = min_max_normalize(query_doc_scores)

        sorted_doc_ids = np.argsort(query_doc_scores)[::-1]
        sorted_doc_scores = query_doc_scores[sorted_doc_ids.tolist()]
        return sorted_doc_ids, sorted_doc_scores


    def get_top_k_weights(self,
                          link_top_k: int,
                          all_phrase_weights: np.ndarray,
                          linking_score_map: Dict[str, float]) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        This function filters the all_phrase_weights to retain only the weights for the
        top-ranked phrases in terms of the linking_score_map. It also filters linking scores
        to retain only the top `link_top_k` ranked nodes. Non-selected phrases in phrase
        weights are reset to a weight of 0.0.

        Args:
            link_top_k (int): Number of top-ranked nodes to retain in the linking score map.
            all_phrase_weights (np.ndarray): An array representing the phrase weights, indexed
                by phrase ID.
            linking_score_map (Dict[str, float]): A mapping of phrase content to its linking
                score, sorted in descending order of scores.

        Returns:
            Tuple[np.ndarray, Dict[str, float]]: A tuple containing the filtered array
            of all_phrase_weights with unselected weights set to 0.0, and the filtered
            linking_score_map containing only the top `link_top_k` phrases.
        """
        # choose top ranked nodes in linking_score_map
        linking_score_map = dict(sorted(linking_score_map.items(), key=lambda x: x[1], reverse=True)[:link_top_k])

        # only keep the top_k phrases in all_phrase_weights
        top_k_phrases = set(linking_score_map.keys())
        top_k_phrases_keys = set(
            [compute_mdhash_id(content=top_k_phrase, prefix="entity-") for top_k_phrase in top_k_phrases])

        for phrase_key in self.node_name_to_vertex_idx:
            if phrase_key not in top_k_phrases_keys:
                phrase_id = self.node_name_to_vertex_idx.get(phrase_key, None)
                if phrase_id is not None:
                    all_phrase_weights[phrase_id] = 0.0

        assert np.count_nonzero(all_phrase_weights) == len(linking_score_map.keys())
        return all_phrase_weights, linking_score_map

    def graph_search_with_fact_entities(self, query: str,
                                        link_top_k: int,
                                        query_fact_scores: np.ndarray,
                                        top_k_facts: List[Tuple],
                                        top_k_fact_indices: List[str],
                                        entity_seed_top_k: Optional[int] = None,
                                        passage_node_weight: float = 0.05,
                                        query_idx: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes document scores based on fact-based similarity and relevance using personalized
        PageRank (PPR). This function starts from relevant fact entities and aggregates
        entity PageRank scores back to documents through entity-to-chunk links.

        Parameters:
            query (str): The input query string for which similarity and relevance computations
                need to be performed.
            link_top_k (int): The number of top phrases to include from the linking score map for
                downstream processing.
            query_fact_scores (np.ndarray): An array of scores representing fact-query similarity
                for each of the provided facts.
            top_k_facts (List[Tuple]): A list of top-ranked facts, where each fact is represented
                as a tuple of its subject, predicate, and object.
            top_k_fact_indices (List[str]): Corresponding indices or identifiers for the top-ranked
                facts in the query_fact_scores array.
            entity_seed_top_k (Optional[int]): Number of entity seed nodes retained for the PPR reset.
                If None, falls back to link_top_k for backward-compatible behavior.
            passage_node_weight (float): Multiplicative factor for DPR passage reset weights.
        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing two arrays:
                - The first array corresponds to document IDs sorted based on their scores.
                - The second array consists of aggregated PPR scores associated with the sorted document IDs.
        """

        #Assigning phrase weights based on selected facts from previous steps.
        linking_score_map = {}  # from phrase to the average scores of the facts that contain the phrase
        phrase_scores = {}  # store all fact scores for each phrase regardless of whether they exist in the knowledge graph or not
        phrase_weights = np.zeros(len(self.graph.vs['name']))
        passage_weights = np.zeros(len(self.graph.vs['name']))
        number_of_occurs = np.zeros(len(self.graph.vs['name']))

        phrases_and_ids = set()

        for rank, f in enumerate(top_k_facts):
            subject_phrase = f[0].lower()
            object_phrase = f[2].lower()
            fact_score = query_fact_scores[
                top_k_fact_indices[rank]] if query_fact_scores.ndim > 0 else query_fact_scores

            for phrase in [subject_phrase, object_phrase]:
                phrase_key = compute_mdhash_id(
                    content=phrase,
                    prefix="entity-"
                )
                phrase_id = self.node_name_to_vertex_idx.get(phrase_key, None)

                if phrase_id is not None:
                    weighted_fact_score = fact_score

                    if len(self.ent_node_to_chunk_ids.get(phrase_key, set())) > 0:
                        weighted_fact_score /= len(self.ent_node_to_chunk_ids[phrase_key])

                    phrase_weights[phrase_id] += weighted_fact_score
                    number_of_occurs[phrase_id] += 1

                    phrases_and_ids.add((phrase, phrase_id))

        phrase_weights = np.divide(
            phrase_weights,
            number_of_occurs,
            out=np.zeros_like(phrase_weights),
            where=number_of_occurs > 0
        )

        for phrase, phrase_id in phrases_and_ids:
            if phrase not in phrase_scores:
                phrase_scores[phrase] = []

            phrase_scores[phrase].append(phrase_weights[phrase_id])

        # calculate average fact score for each phrase
        for phrase, scores in phrase_scores.items():
            linking_score_map[phrase] = float(np.mean(scores))

        if entity_seed_top_k is None:
            entity_seed_top_k = link_top_k

        if entity_seed_top_k:
            phrase_weights, linking_score_map = self.get_top_k_weights(entity_seed_top_k,
                                                                           phrase_weights,
                                                                           linking_score_map)  # at this stage, the entity seed count is determined by entity_seed_top_k

        # Add a weak DPR passage prior to the PPR reset vector, matching the
        # baseline mechanism while keeping passage edges purely structural.
        dpr_sorted_doc_ids, dpr_sorted_doc_scores = self.dense_passage_retrieval(query)
        normalized_dpr_sorted_scores = min_max_normalize(dpr_sorted_doc_scores)

        # Vectorized passage_weights assignment using the precomputed
        # chunk_idx -> graph vertex idx lookup (see _precompute_retrieval_caches).
        v_ids = self._cached_chunk_to_passage_v_idx[dpr_sorted_doc_ids]
        valid_mask = v_ids >= 0
        if valid_mask.any():
            passage_weights[v_ids[valid_mask]] = (
                normalized_dpr_sorted_scores[valid_mask] * passage_node_weight
            )

        # linking_score_map only needs the top entries (later truncated to 30 below);
        # restrict the get_row text fetches to the top 50 DPR results to avoid an
        # O(n_chunks) parquet scan per query.
        text_top_k = min(50, len(dpr_sorted_doc_ids))
        dpr_ids_head = dpr_sorted_doc_ids[:text_top_k].tolist()
        v_ids_head = v_ids[:text_top_k].tolist()
        scores_head = normalized_dpr_sorted_scores[:text_top_k].tolist()
        for i, doc_id in enumerate(dpr_ids_head):
            if v_ids_head[i] < 0:
                continue
            passage_node_key = self.passage_node_keys[doc_id]
            passage_node_text = self.chunk_embedding_store.get_row(passage_node_key)["content"]
            linking_score_map[passage_node_text] = float(scores_head[i]) * passage_node_weight

        node_weights = phrase_weights + passage_weights

        #Recording top 30 facts in linking_score_map
        if len(linking_score_map) > 30:
            linking_score_map = dict(sorted(linking_score_map.items(), key=lambda x: x[1], reverse=True)[:30])

        assert sum(node_weights) > 0, f'No phrases found in the graph for the given facts: {top_k_facts}'

        # Probability black-hole experiment: optionally capture per-query rel_* and
        # full pagerank_scores into a local dict (thread-safe, no algorithm change).
        black_hole_capture = {} if getattr(self.global_config, "dump_entity_black_hole", False) else None

        # Diversity-guided directed PPR: rewrite per-edge weights based on the current query.
        if self.global_config.enable_directed_ppr:
            edge_weights = self.build_directed_edge_weights(query, capture=black_hole_capture)
        else:
            edge_weights = None

        #Running PPR algorithm based on the phrase weights previously assigned
        ppr_start = time.time()
        ppr_sorted_doc_ids, ppr_sorted_doc_scores = self.run_ppr(
            node_weights,
            damping=self.global_config.damping,
            edge_weights=edge_weights,
            capture=black_hole_capture,
        )
        ppr_end = time.time()

        with self._timing_lock:
            self.ppr_time += (ppr_end - ppr_start)

        assert len(ppr_sorted_doc_ids) == len(
            self.chunk_keys), f"Doc prob length {len(ppr_sorted_doc_ids)} != corpus length {len(self.chunk_keys)}"

        if getattr(self.global_config, "dump_retrieval_artifacts", False):
            try:
                self._dump_retrieval_artifact(
                    query=query,
                    query_idx=query_idx,
                    phrase_weights=phrase_weights,
                    dpr_sorted_doc_ids=dpr_sorted_doc_ids,
                    dpr_raw_scores=dpr_sorted_doc_scores,
                    dpr_normalized_scores=normalized_dpr_sorted_scores,
                    passage_node_weight=passage_node_weight,
                    ppr_sorted_doc_ids=ppr_sorted_doc_ids,
                    ppr_sorted_doc_scores=ppr_sorted_doc_scores,
                )
            except Exception as exc:
                logger.warning(f"Failed to dump retrieval artifact for query_idx={query_idx}: {exc}")

        if black_hole_capture is not None:
            try:
                self._dump_entity_black_hole_artifact(
                    query=query,
                    query_idx=query_idx,
                    capture=black_hole_capture,
                )
            except Exception as exc:
                logger.warning(f"Failed to dump entity black-hole artifact for query_idx={query_idx}: {exc}")

        if getattr(self.global_config, "ppr_experiment_mode", "none") != "none":
            try:
                self._run_entity_only_ppr_experiment(
                    query=query,
                    query_idx=query_idx,
                    phrase_weights=phrase_weights,
                    capture=black_hole_capture,
                )
            except Exception as exc:
                logger.warning(f"Entity-only PPR experiment failed q_idx={query_idx}: {exc}")

        return ppr_sorted_doc_ids, ppr_sorted_doc_scores


    def _dump_retrieval_artifact(self,
                                 query: str,
                                 query_idx: Optional[int],
                                 phrase_weights: np.ndarray,
                                 dpr_sorted_doc_ids: np.ndarray,
                                 dpr_raw_scores: np.ndarray,
                                 dpr_normalized_scores: np.ndarray,
                                 passage_node_weight: float,
                                 ppr_sorted_doc_ids: np.ndarray,
                                 ppr_sorted_doc_scores: np.ndarray) -> None:
        """Append one JSONL line per query: entity seeds, DPR passage seeds, top-K PPR passages.

        Schema (per line):
          query_idx, query,
          entity_seeds:  [{rank, node_idx, name, weight}]                  (nonzero phrase_weights, sorted desc)
          passage_seeds: [{rank, node_idx, passage_key, passage_chunk_id,
                           dpr_score, dpr_score_normalized, weighted}]      (top retrieval_artifacts_top_k_seed)
          top_k_passages:[{rank, node_idx, passage_key, passage_chunk_id,
                           ppr_score}]                                      (top retrieval_artifacts_top_k_passage)
        """
        cfg = self.global_config
        top_k_passage = int(getattr(cfg, "retrieval_artifacts_top_k_passage", 100))
        top_k_seed = int(getattr(cfg, "retrieval_artifacts_top_k_seed", 5))

        vertex_names = self.graph.vs['name']

        nz_idx = np.flatnonzero(phrase_weights > 0)
        if nz_idx.size > 0:
            order = nz_idx[np.argsort(-phrase_weights[nz_idx])]
        else:
            order = nz_idx
        entity_seeds = [
            {
                "rank": int(r),
                "node_idx": int(idx),
                "name": vertex_names[int(idx)],
                "weight": float(phrase_weights[int(idx)]),
            }
            for r, idx in enumerate(order.tolist())
        ]

        def _to_list(arr):
            return arr.tolist() if hasattr(arr, "tolist") else list(arr)

        dpr_ids_list = _to_list(dpr_sorted_doc_ids)
        dpr_raw_list = _to_list(dpr_raw_scores)
        dpr_norm_list = _to_list(dpr_normalized_scores)

        passage_seeds = []
        for rank, doc_id in enumerate(dpr_ids_list[:top_k_seed]):
            passage_node_key = self.passage_node_keys[doc_id]
            node_idx = self.node_name_to_vertex_idx.get(passage_node_key, None)
            passage_seeds.append({
                "rank": int(rank),
                "node_idx": -1 if node_idx is None else int(node_idx),
                "passage_key": passage_node_key,
                "passage_chunk_id": int(doc_id),
                "dpr_score": float(dpr_raw_list[rank]),
                "dpr_score_normalized": float(dpr_norm_list[rank]),
                "weighted": float(dpr_norm_list[rank]) * float(passage_node_weight),
            })

        ppr_ids_list = _to_list(ppr_sorted_doc_ids)
        ppr_scores_list = _to_list(ppr_sorted_doc_scores)
        top_k_passages = []
        for rank, doc_id in enumerate(ppr_ids_list[:top_k_passage]):
            passage_node_key = self.passage_node_keys[doc_id]
            node_idx = self.node_name_to_vertex_idx.get(passage_node_key, None)
            top_k_passages.append({
                "rank": int(rank),
                "node_idx": -1 if node_idx is None else int(node_idx),
                "passage_key": passage_node_key,
                "passage_chunk_id": int(doc_id),
                "ppr_score": float(ppr_scores_list[rank]),
            })

        record = {
            "query_idx": -1 if query_idx is None else int(query_idx),
            "query": query,
            "entity_seeds": entity_seeds,
            "passage_seeds": passage_seeds,
            "top_k_passages": top_k_passages,
        }

        path = os.path.join(self.global_config.save_dir, "retrieval_artifacts.jsonl")
        line = json.dumps(record, ensure_ascii=False)
        with self._artifact_lock:
            os.makedirs(self.global_config.save_dir, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")


    def _dump_entity_black_hole_artifact(self,
                                         query: str,
                                         query_idx: Optional[int],
                                         capture: dict) -> None:
        """Record top-K entity nodes by converged PPR score for the black-hole experiment.

        Each query produces one JSONL line containing the top entities ranked by
        pagerank_scores_full (restricted to entity vertices via _cached_node_is_entity),
        each with diversity_raw / diversity_norm (query-independent) and
        rel_raw / rel_norm (query-specific, populated by build_directed_edge_weights).

        Side-channel only: relies on `capture` populated by build_directed_edge_weights
        and run_ppr, no algorithm path is altered.
        """
        cfg = self.global_config
        top_k = int(getattr(cfg, "entity_black_hole_top_k", 100))

        pr = capture.get('pagerank_scores_full', None)
        if pr is None:
            return
        pr = np.asarray(pr, dtype=np.float64)
        n_nodes = pr.shape[0]

        entity_mask = getattr(self, '_cached_node_is_entity', None)
        if entity_mask is None or entity_mask.shape[0] != n_nodes:
            return
        entity_indices = np.flatnonzero(entity_mask)
        if entity_indices.size == 0:
            return

        pr_entity = pr[entity_indices]
        order = entity_indices[np.argsort(-pr_entity)][:top_k]

        rel_raw = capture.get('rel_raw', None)
        rel_norm = capture.get('rel_norm', None)

        div_raw = getattr(self, 'entity_diversity_raw', None)
        if div_raw is None or len(div_raw) != n_nodes:
            div_raw = np.zeros(n_nodes, dtype=np.float32)
        div_norm = getattr(self, 'entity_diversity_norm', None)
        if div_norm is None or len(div_norm) != n_nodes:
            div_norm = getattr(self, '_cached_div_norm', None)
            if div_norm is None or len(div_norm) != n_nodes:
                div_norm = np.full(n_nodes, 0.5, dtype=np.float32)

        vertex_names = getattr(self, '_cached_vertex_names', None) or self.graph.vs['name']

        top_entities = []
        for rank, idx in enumerate(order.tolist()):
            idx_i = int(idx)
            top_entities.append({
                "rank": int(rank),
                "node_idx": idx_i,
                "name": vertex_names[idx_i],
                "ppr_score": float(pr[idx_i]),
                "diversity_raw": float(div_raw[idx_i]),
                "diversity_norm": float(div_norm[idx_i]),
                "rel_raw": float(rel_raw[idx_i]) if rel_raw is not None else None,
                "rel_norm": float(rel_norm[idx_i]) if rel_norm is not None else None,
            })

        record = {
            "query_idx": -1 if query_idx is None else int(query_idx),
            "query": query,
            "damping": float(cfg.damping),
            "reset_prob": float(1.0 - cfg.damping),
            "top_k_entities": top_entities,
        }

        path = os.path.join(cfg.save_dir, "entity_black_hole_artifacts.jsonl")
        line = json.dumps(record, ensure_ascii=False)
        with self._artifact_lock:
            os.makedirs(cfg.save_dir, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")


    def _compute_entity_only_directed_weights(self, eo_graph, eo_full, rel_norm_full):
        """Compute Innovation directed-PPR edge weights on the entity-only subgraph.

        Mirrors the entity-entity branch of build_directed_edge_weights:
          score = lambda_rel * rel_norm[v] + lambda_prior * prior_norm[e] - lambda_div * |delta|
          + 9:1 down/up direction split with within-direction normalization
          + A+ self-loop absorption at down-dead nodes
        Passage edges do not exist here, so passage-edge branches are dropped.
        Side-channel; does not touch the main retrieval path.
        """
        cfg = self.global_config
        n_eo = eo_graph.vcount()
        n_eo_edges = eo_graph.ecount()
        if n_eo_edges == 0:
            return []

        edgelist_eo = np.asarray(eo_graph.get_edgelist(), dtype=np.int64)
        u_eo = edgelist_eo[:, 0]
        v_eo = edgelist_eo[:, 1]
        is_self_loop = u_eo == v_eo

        div_norm_eo = (
            np.asarray(eo_graph.vs['diversity_norm'], dtype=np.float32)
            if 'diversity_norm' in eo_graph.vs.attributes()
            else np.full(n_eo, 0.5, dtype=np.float32)
        )
        prior_norm_eo = (
            np.asarray([0.0 if v is None else v for v in eo_graph.es['prior_norm']], dtype=np.float32)
            if 'prior_norm' in eo_graph.es.attributes()
            else np.zeros(n_eo_edges, dtype=np.float32)
        )

        if rel_norm_full is not None and len(rel_norm_full) == self.graph.vcount():
            rel_norm_eo = np.asarray(rel_norm_full, dtype=np.float32)[eo_full]
        else:
            rel_norm_eo = np.zeros(n_eo, dtype=np.float32)

        delta = div_norm_eo[v_eo] - div_norm_eo[u_eo]
        prior_term = cfg.ppr_score_lambda_prior * prior_norm_eo if cfg.ppr_use_prior_edge_weight else 0.0
        score_raw = (
            cfg.ppr_score_const_base
            + cfg.ppr_score_lambda_rel * rel_norm_eo[v_eo]
            + prior_term
            - cfg.ppr_score_lambda_div * np.abs(delta)
        )
        score = np.maximum(score_raw, cfg.ppr_score_epsilon).astype(np.float64)

        if cfg.ppr_skip_direction_control:
            weights_a1 = np.zeros(n_eo_edges, dtype=np.float64)
            non_self = ~is_self_loop
            weights_a1[non_self] = score[non_self]
            return weights_a1.tolist()

        score[is_self_loop] = 0.0
        is_down = (delta <= 0) & ~is_self_loop
        is_up = (delta > 0) & ~is_self_loop
        S_D = np.bincount(u_eo[is_down], weights=score[is_down], minlength=n_eo).astype(np.float64)
        S_U = np.bincount(u_eo[is_up], weights=score[is_up], minlength=n_eo).astype(np.float64)

        node_down_dead = S_D == 0.0
        node_up_dead = S_U == 0.0
        nonself_out_deg = np.bincount(u_eo[~is_self_loop], minlength=n_eo).astype(np.int64)
        node_isolated = nonself_out_deg == 0

        p_down = float(cfg.ppr_down_direction_prob)
        p_up = float(cfg.ppr_up_direction_prob)
        eff_down = np.where(node_up_dead & ~node_down_dead, p_down + p_up, p_down)
        if cfg.ppr_self_loop_at_local_min:
            eff_up = np.full(n_eo, p_up, dtype=np.float64)
        else:
            eff_up = np.where(node_down_dead & ~node_up_dead, p_down + p_up, p_up)

        safe_S_D = np.where(S_D > 0, S_D, 1.0)
        safe_S_U = np.where(S_U > 0, S_U, 1.0)
        weights = np.zeros(n_eo_edges, dtype=np.float64)
        weights[is_down] = (eff_down[u_eo] * score / safe_S_D[u_eo])[is_down]
        weights[is_up] = (eff_up[u_eo] * score / safe_S_U[u_eo])[is_up]

        sl_node_weight = np.zeros(n_eo, dtype=np.float64)
        if cfg.ppr_self_loop_at_local_min:
            sl_node_weight[node_down_dead & ~node_up_dead] = p_down
        sl_node_weight[node_isolated] = 1.0
        if is_self_loop.any():
            weights[is_self_loop] = sl_node_weight[u_eo[is_self_loop]]
        return weights.tolist()


    def _run_entity_only_ppr_experiment(self, query, query_idx, phrase_weights, capture):
        """Side-channel: run PPR on entity-only subgraph in either innovation or vanilla mode,
        record top-K entity div/sim/ppr to a mode-suffixed JSONL.

        Does NOT alter the main retrieval pipeline. Reads only:
          - self._eo_graph                 (precomputed entity-only subgraph)
          - self._eo_full_v_indices        (eo_idx -> full_idx mapping)
          - capture['rel_norm'] (optional) (per-query, populated by build_directed_edge_weights)
        """
        cfg = self.global_config
        mode = getattr(cfg, "ppr_experiment_mode", "none")
        if mode == "none" or self._eo_graph is None:
            return
        if mode not in ("entity_only_innovation", "entity_only_vanilla"):
            logger.warning(f"Unknown ppr_experiment_mode={mode!r}, skipping.")
            return

        eo_graph = self._eo_graph
        n_eo = eo_graph.vcount()
        if n_eo == 0:
            return

        eo_full = self._eo_full_v_indices
        reset_eo = np.asarray(phrase_weights[eo_full], dtype=np.float64).copy()
        reset_eo = np.where(np.isnan(reset_eo) | (reset_eo < 0), 0.0, reset_eo)
        if reset_eo.sum() <= 0:
            return

        if mode == "entity_only_vanilla":
            edge_weights_eo = 'weight'
        else:
            rel_norm_full = capture.get('rel_norm', None) if capture else None
            edge_weights_eo = self._compute_entity_only_directed_weights(
                eo_graph, eo_full, rel_norm_full,
            )

        pr_eo = eo_graph.personalized_pagerank(
            vertices=range(n_eo),
            damping=cfg.damping,
            directed=cfg.is_directed_graph,
            weights=edge_weights_eo,
            reset=reset_eo.tolist(),
            implementation='prpack',
        )
        pr_eo = np.asarray(pr_eo, dtype=np.float64)

        top_k = int(getattr(cfg, "entity_black_hole_top_k", 100))
        order_eo = np.argsort(-pr_eo)[:top_k]
        full_top = eo_full[order_eo]

        rel_norm = capture.get('rel_norm', None) if capture else None
        div_norm = getattr(self, 'entity_diversity_norm', None)
        n_full = self.graph.vcount()
        if div_norm is None or len(div_norm) != n_full:
            div_norm = getattr(self, '_cached_div_norm', None)
            if div_norm is None or len(div_norm) != n_full:
                div_norm = np.full(n_full, 0.5, dtype=np.float32)
        vertex_names = getattr(self, '_cached_vertex_names', None) or self.graph.vs['name']

        top_entities = []
        for rank, (eo_idx, full_idx) in enumerate(zip(order_eo.tolist(), full_top.tolist())):
            top_entities.append({
                "rank": int(rank),
                "node_idx": int(full_idx),
                "eo_node_idx": int(eo_idx),
                "name": vertex_names[int(full_idx)],
                "ppr_score": float(pr_eo[int(eo_idx)]),
                "diversity_norm": float(div_norm[int(full_idx)]),
                "rel_norm": float(rel_norm[int(full_idx)]) if rel_norm is not None else None,
            })

        record = {
            "query_idx": -1 if query_idx is None else int(query_idx),
            "query": query,
            "ppr_experiment_mode": mode,
            "damping": float(cfg.damping),
            "reset_prob": float(1.0 - cfg.damping),
            "eo_vcount": int(n_eo),
            "eo_ecount": int(eo_graph.ecount()),
            "top_k_entities": top_entities,
        }
        path = os.path.join(cfg.save_dir, f"entity_only_artifacts_{mode}.jsonl")
        line = json.dumps(record, ensure_ascii=False)
        with self._artifact_lock:
            os.makedirs(cfg.save_dir, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")


    def rerank_facts(self, query: str, query_fact_scores: np.ndarray) -> Tuple[List[int], List[Tuple], dict]:
        """

        Args:

        Returns:
            top_k_fact_indicies:
            top_k_facts:
            rerank_log (dict): {'facts_before_rerank': candidate_facts, 'facts_after_rerank': top_k_facts}
                - candidate_facts (list): list of link_top_k facts (each fact is a relation triple in tuple data type).
                - top_k_facts:


        """
        # load args
        link_top_k: int = self.global_config.linking_top_k
        
        # Check if there are any facts to rerank
        if len(query_fact_scores) == 0 or len(self.fact_node_keys) == 0:
            logger.warning("No facts available for reranking. Returning empty lists.")
            return [], [], {'facts_before_rerank': [], 'facts_after_rerank': []}
            
        try:
            # Get the top k facts by score
            if len(query_fact_scores) <= link_top_k:
                # If we have fewer facts than requested, use all of them
                candidate_fact_indices = np.argsort(query_fact_scores)[::-1].tolist()
            else:
                # Otherwise get the top k
                candidate_fact_indices = np.argsort(query_fact_scores)[-link_top_k:][::-1].tolist()
                
            # Get the actual fact IDs
            real_candidate_fact_ids = [self.fact_node_keys[idx] for idx in candidate_fact_indices]
            fact_row_dict = self.fact_embedding_store.get_rows(real_candidate_fact_ids)
            candidate_facts = [eval(fact_row_dict[id]['content']) for id in real_candidate_fact_ids]

            if self.global_config.skip_fact_rerank:
                rerank_log = {
                    'facts_before_rerank': candidate_facts,
                    'facts_after_rerank': candidate_facts,
                    'skip_fact_rerank': True,
                }
                return candidate_fact_indices, candidate_facts, rerank_log
            
            # Rerank the facts
            top_k_fact_indices, top_k_facts, reranker_dict = self.rerank_filter(query,
                                                                                candidate_facts,
                                                                                candidate_fact_indices,
                                                                                len_after_rerank=link_top_k)
            
            rerank_log = {'facts_before_rerank': candidate_facts, 'facts_after_rerank': top_k_facts}
            
            return top_k_fact_indices, top_k_facts, rerank_log
            
        except Exception as e:
            logger.error(f"Error in rerank_facts: {str(e)}")
            return [], [], {'facts_before_rerank': [], 'facts_after_rerank': [], 'error': str(e)}
    
    def _ensure_self_loops(self):
        """
        Add a placeholder self-loop edge for every vertex if one does not exist.

        Used by directed-PPR to host mass absorption at diversity local minima (S_D = 0)
        and the isolated-node fallback. Placeholder weight = 0 so it is inert when the
        source node has well-populated down / up buckets.
        """
        n_nodes = self.graph.vcount()
        if n_nodes == 0:
            return

        edgelist = self.graph.get_edgelist()
        existing = {u for u, v in edgelist if u == v}
        to_add = [(i, i) for i in range(n_nodes) if i not in existing]
        if not to_add:
            return

        attrs = {'weight': [0.0] * len(to_add)}
        existing_es_attrs = self.graph.es.attributes() if self.graph.ecount() > 0 else []
        if 'prior_norm' in existing_es_attrs:
            attrs['prior_norm'] = [0.0] * len(to_add)
        self.graph.add_edges(to_add, attributes=attrs)
        logger.info(f"Added {len(to_add)} self-loop placeholders for directed PPR.")

    def precompute_prior_norm(self):
        """
        Normalize the current `graph.es['weight']` (fact + synonymy mixture) into a
        percentile-truncated Min-Max [0, 1] value stored on `graph.es['prior_norm']`.
        Self-loop placeholders are excluded from the percentile estimation and pinned to 0.
        """
        n_edges = self.graph.ecount()
        if n_edges == 0:
            self.graph.es['prior_norm'] = []
            return

        weights = np.asarray(self.graph.es['weight'], dtype=np.float64)
        edgelist = np.asarray(self.graph.get_edgelist(), dtype=np.int64)
        is_self_loop = edgelist[:, 0] == edgelist[:, 1]
        chunk_ids = set(getattr(self, 'chunk_keys', self.chunk_embedding_store.get_all_ids()))
        vertex_names = self.graph.vs['name']
        src_is_passage = np.asarray([vertex_names[src] in chunk_ids for src in edgelist[:, 0]], dtype=bool)
        dst_is_passage = np.asarray([vertex_names[dst] in chunk_ids for dst in edgelist[:, 1]], dtype=bool)
        is_passage_edge = (src_is_passage | dst_is_passage) & ~is_self_loop

        prior_norm = np.zeros(n_edges, dtype=np.float32)
        real_mask = ~is_self_loop & ~is_passage_edge
        if real_mask.any():
            real_norm = truncated_min_max_normalize(
                weights[real_mask],
                lo_q=self.global_config.ppr_truncation_low,
                hi_q=self.global_config.ppr_truncation_high,
            )
            prior_norm[real_mask] = real_norm
        self.graph.es['prior_norm'] = prior_norm.tolist()
        logger.info(f"Computed prior_norm over {int(real_mask.sum())} non-passage, non-self-loop edges.")

    def precompute_entity_diversity(self):
        """
        For each entity vertex, diversity = trace(cov) of the embeddings of the passages the entity
        appears in = mean(||x||^2) - ||centroid||^2. For L2-normalized embeddings (NV-Embed-v2 etc.)
        this reduces to 1 - ||centroid||^2.

        Results:
          - self.entity_diversity_raw : shape (n_nodes,), pre-normalization values
          - self.entity_diversity_norm : shape (n_nodes,), after 1%-99% truncated Min-Max
          - self.graph.vs['diversity_norm'] persisted so reload without re-indexing keeps the values
        """
        n_nodes = self.graph.vcount()
        if n_nodes == 0:
            self.entity_diversity_raw = np.zeros(0, dtype=np.float32)
            self.entity_diversity_norm = np.zeros(0, dtype=np.float32)
            self.graph.vs['diversity_norm'] = []
            return

        if self.ent_node_to_chunk_ids is None:
            logger.warning("ent_node_to_chunk_ids is None; diversity set to zeros.")
            self.entity_diversity_raw = np.zeros(n_nodes, dtype=np.float32)
            self.entity_diversity_norm = np.full(n_nodes, 0.5, dtype=np.float32)
            self.graph.vs['diversity_norm'] = self.entity_diversity_norm.tolist()
            return

        div_raw = np.zeros(n_nodes, dtype=np.float32)
        div_norm = np.full(n_nodes, 0.5, dtype=np.float32)
        vertex_names = self.graph.vs['name']
        chunk_ids = set(getattr(self, 'chunk_keys', self.chunk_embedding_store.get_all_ids()))
        is_entity_node = np.asarray([node_name not in chunk_ids for node_name in vertex_names], dtype=bool)
        for v_idx, ent_key in enumerate(tqdm(vertex_names, desc="Computing entity diversity")):
            if not is_entity_node[v_idx]:
                continue
            ck_set = self.ent_node_to_chunk_ids.get(ent_key, None)
            if not ck_set or len(ck_set) < 2:
                # Single (or zero) passage → variance is 0 by definition.
                continue
            ck_list = list(ck_set)
            chunk_embs = self.chunk_embedding_store.get_embeddings(ck_list)
            if chunk_embs is None or len(chunk_embs) == 0:
                continue
            chunk_embs = np.asarray(chunk_embs, dtype=np.float64)
            centroid = chunk_embs.mean(axis=0)
            div_raw[v_idx] = float((chunk_embs ** 2).sum(axis=1).mean() - (centroid ** 2).sum())

        self.entity_diversity_raw = div_raw
        if is_entity_node.any():
            div_norm[is_entity_node] = truncated_min_max_normalize(
                div_raw[is_entity_node],
                lo_q=self.global_config.ppr_truncation_low,
                hi_q=self.global_config.ppr_truncation_high,
            )
        self.entity_diversity_norm = div_norm
        self.graph.vs['diversity_norm'] = self.entity_diversity_norm.tolist()
        logger.info(
            f"Computed entity diversity. Raw mean={div_raw[is_entity_node].mean() if is_entity_node.any() else 0.0:.4f}, "
            f"non-zero fraction={(div_raw[is_entity_node] > 0).mean() if is_entity_node.any() else 0.0:.3f}."
        )

    def build_directed_edge_weights(self, query: str, capture: Optional[dict] = None):
        """
        Per-query rewrite of graph.es['weight'] implementing diversity-guided directed PPR.

        For each directed edge u -> v:
            delta   = div_norm[v] - div_norm[u]
            score   = lambda_r * rel_norm[v] + lambda_p * prior_norm[e] - lambda_d * |delta|
            score+  = max(epsilon, score)                                        (ReLU + floor)
            bucket  = down if delta <= 0 else up                                  (self-loops excluded)

        Per source node u, within-direction normalization yields row sum 1.0:
            w(u->v) = down_prob * score+ / S_D(u)   if down
            w(u->v) = up_prob   * score+ / S_U(u)   if up

        Degenerate fallback:
            S_D(u) = 0 only  -> self-loop with weight down_prob (A+)  if enabled,
                                else merge down budget into up (pure A).
            S_U(u) = 0 only  -> merge up budget into down.
            both  = 0        -> self-loop weight 1.0 if placeholder exists.

        Pre-conditions: `_ensure_self_loops`, `precompute_prior_norm`,
        `precompute_entity_diversity`, and `get_query_embeddings([query])` all ran.
        """
        cfg = self.global_config
        n_nodes = self.graph.vcount()
        n_edges = self.graph.ecount()
        if n_nodes == 0 or n_edges == 0:
            return

        # ===== Query-dependent per-entity relevance (max-pool DPR over entity's passages) =====
        query_emb = self.query_to_embedding['passage'].get(query, None)
        if query_emb is None:
            query_emb = self.embedding_model.batch_encode(
                query,
                instruction=get_query_instruction('query_to_passage'),
                norm=True,
            )
        query_emb = np.asarray(query_emb, dtype=np.float32).reshape(-1)

        passage_scores = self.passage_embeddings @ query_emb  # shape (n_chunks,)

        # Read precomputed (query-independent) caches built in _precompute_retrieval_caches.
        node_is_passage = self._cached_node_is_passage
        node_is_entity = self._cached_node_is_entity

        # Vectorized rel_raw: gather passage scores for each entity's chunk set,
        # then segment-max via reduceat. Entities without chunks stay at 0,
        # matching the original `if not ck_set: continue` semantics.
        rel_raw = np.zeros(n_nodes, dtype=np.float32)
        rel_norm = np.zeros(n_nodes, dtype=np.float32)
        if self._entity_v_idx_with_chunks.size > 0:
            chunk_scores = passage_scores[self._entity_chunk_indices].astype(np.float32, copy=False)
            seg_max = np.maximum.reduceat(chunk_scores, self._entity_chunk_indptr[:-1])
            rel_raw[self._entity_v_idx_with_chunks] = seg_max

        if node_is_entity.any():
            rel_norm[node_is_entity] = truncated_min_max_normalize(
                rel_raw[node_is_entity], lo_q=cfg.ppr_truncation_low, hi_q=cfg.ppr_truncation_high
            )

        # ===== Cached structural signals =====
        prior_norm = self._cached_prior_norm

        div_norm = getattr(self, 'entity_diversity_norm', None)
        if div_norm is None or len(div_norm) != n_nodes:
            div_norm = self._cached_div_norm

        # ===== Edge-wise score (edge masks/index arrays from cache) =====
        u_idx = self._cached_edgelist_u
        v_idx = self._cached_edgelist_v
        is_self_loop = self._cached_is_self_loop
        is_passage_edge = self._cached_is_passage_edge
        is_entity_edge = self._cached_is_entity_edge

        delta = div_norm[v_idx] - div_norm[u_idx]
        prior_term = cfg.ppr_score_lambda_prior * prior_norm if cfg.ppr_use_prior_edge_weight else 0.0
        score_raw = (
            cfg.ppr_score_const_base
            + cfg.ppr_score_lambda_rel * rel_norm[v_idx]
            + prior_term
            - cfg.ppr_score_lambda_div * np.abs(delta)
        )
        score = np.maximum(score_raw, cfg.ppr_score_epsilon).astype(np.float64)

        # ---- A1 ablation: skip the 9:1 directional control ----
        # Use the score formula's output as the raw entity-entity edge weight, treat
        # passage edges as structural connectors (1.0), and disable self-loop absorption.
        # PPR still runs on the directed graph, so u->v and v->u may have different
        # raw scores.
        if cfg.ppr_skip_direction_control:
            weights_a1 = np.zeros(n_edges, dtype=np.float64)
            weights_a1[is_entity_edge] = score[is_entity_edge]
            weights_a1[is_passage_edge] = 1.0
            # Self-loops kept at 0 → no mass absorption, no degenerate fallback.
            if capture is not None:
                capture['rel_raw'] = rel_raw.copy()
                capture['rel_norm'] = rel_norm.copy()
            return weights_a1.tolist()

        # Only entity-entity edges participate in the directed diversity walk.
        score[~is_entity_edge] = 0.0

        is_down = (delta <= 0) & is_entity_edge
        is_up = (delta > 0) & is_entity_edge

        # Per-source direction sums.
        S_D = np.bincount(u_idx[is_down], weights=score[is_down], minlength=n_nodes).astype(np.float64)
        S_U = np.bincount(u_idx[is_up], weights=score[is_up], minlength=n_nodes).astype(np.float64)

        node_down_dead = (S_D == 0.0) & node_is_entity
        node_up_dead = (S_U == 0.0) & node_is_entity
        nonself_out_degree = np.bincount(u_idx[~is_self_loop], minlength=n_nodes).astype(np.int64)
        node_isolated = nonself_out_degree == 0

        p_down = float(cfg.ppr_down_direction_prob)
        p_up = float(cfg.ppr_up_direction_prob)

        # Effective direction budgets per source node.
        # - If up is dead and down is alive: down absorbs the up budget (option A for up-side).
        # - If down is dead and up is alive: self-loop eats the down budget (A+); otherwise merge into up.
        eff_down = np.where(node_up_dead & ~node_down_dead, p_down + p_up, p_down)
        if cfg.ppr_self_loop_at_local_min:
            eff_up = np.full(n_nodes, p_up, dtype=np.float64)
        else:
            eff_up = np.where(node_down_dead & ~node_up_dead, p_down + p_up, p_up)

        safe_S_D = np.where(S_D > 0, S_D, 1.0)
        safe_S_U = np.where(S_U > 0, S_U, 1.0)

        weights = np.zeros(n_edges, dtype=np.float64)
        weights_down_all = eff_down[u_idx] * score / safe_S_D[u_idx]
        weights_up_all = eff_up[u_idx] * score / safe_S_U[u_idx]
        weights[is_down] = weights_down_all[is_down]
        weights[is_up] = weights_up_all[is_up]
        # Passage edges are structural connectors only: no query, diversity, or semantic prior weighting.
        weights[is_passage_edge] = 1.0

        # Self-loop weights per source node.
        sl_node_weight = np.zeros(n_nodes, dtype=np.float64)
        if cfg.ppr_self_loop_at_local_min:
            # Local min absorption: down-dead and up-alive.
            sl_node_weight[node_down_dead & ~node_up_dead] = p_down
        # Isolated nodes absorb the full walking budget so prpack sees a non-zero row.
        sl_node_weight[node_isolated] = 1.0

        if is_self_loop.any():
            sl_edge_src = u_idx[is_self_loop]
            weights[is_self_loop] = sl_node_weight[sl_edge_src]

        if capture is not None:
            capture['rel_raw'] = rel_raw.copy()
            capture['rel_norm'] = rel_norm.copy()
        return weights.tolist()

    def run_ppr(self,
                reset_prob: np.ndarray,
                damping: float =0.5,
                edge_weights: Optional[List[float]] = None,
                capture: Optional[dict] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs Personalized PageRank (PPR) on a graph and computes relevance scores for
        retrievable passage nodes. The method utilizes a damping
        factor for teleportation during rank computation and can take a reset
        probability array to influence the starting state of the computation.

        Parameters:
            reset_prob (np.ndarray): A 1-dimensional array specifying the reset
                probability distribution for each node. The array must have a size
                equal to the number of nodes in the graph. NaNs or negative values
                within the array are replaced with zeros.
            damping (float): A scalar specifying the damping factor for the
                computation. Defaults to 0.5 if not provided or set to `None`.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing two numpy arrays. The
            first array represents the sorted passage IDs based on their
            PageRank scores in descending order. The second array contains the
            corresponding passage scores in the same order.
        """

        if damping is None: damping = 0.5 # for potential compatibility
        reset_prob = np.where(np.isnan(reset_prob) | (reset_prob < 0), 0, reset_prob)
        pagerank_scores = self.graph.personalized_pagerank(
            vertices=range(len(self.node_name_to_vertex_idx)),
            damping=damping,
            directed=self.global_config.is_directed_graph,
            weights=edge_weights if edge_weights is not None else 'weight',
            reset=reset_prob,
            implementation='prpack'
        )

        if capture is not None:
            capture['pagerank_scores_full'] = np.asarray(pagerank_scores, dtype=np.float64)

        doc_scores = np.array([pagerank_scores[idx] for idx in self.passage_node_idxs])
        sorted_doc_ids = np.argsort(doc_scores)[::-1]
        sorted_doc_scores = doc_scores[sorted_doc_ids.tolist()]

        return sorted_doc_ids, sorted_doc_scores
