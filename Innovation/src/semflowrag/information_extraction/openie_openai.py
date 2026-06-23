import ast
import json
import re
from dataclasses import dataclass
from typing import Dict, Any, List, TypedDict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from ..prompts import PromptTemplateManager
from ..utils.logging_utils import get_logger
from ..utils.llm_utils import fix_broken_generated_json, filter_invalid_triples
from ..utils.misc_utils import TripleRawOutput, NerRawOutput
from ..llm.openai_gpt import CacheOpenAI

logger = get_logger(__name__)


def _strip_markdown_fence(text: str) -> str:
    match = re.match(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", text, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text.strip()


def _extract_balanced_segment(text: str, start_idx: int, open_char: str, close_char: str) -> str:
    depth = 0
    in_string = False
    quote_char = ""
    escaped = False

    for idx in range(start_idx, len(text)):
        char = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote_char:
                in_string = False
            continue

        if char in ('"', "'"):
            in_string = True
            quote_char = char
        elif char == open_char:
            depth += 1
        elif char == close_char:
            depth -= 1
            if depth == 0:
                return text[start_idx:idx + 1]

    return ""


def _iter_json_object_candidates(text: str) -> List[str]:
    candidates = []
    cleaned = _strip_markdown_fence(text)
    if cleaned:
        candidates.append(cleaned)

    for idx, char in enumerate(cleaned):
        if char == "{":
            candidate = _extract_balanced_segment(cleaned, idx, "{", "}")
            if candidate:
                candidates.append(candidate)

    return candidates


def _iter_keyed_array_candidates(text: str, expected_keys: Tuple[str, ...]) -> List[str]:
    candidates = []
    cleaned = _strip_markdown_fence(text)
    for key in expected_keys:
        key_pattern = re.compile(rf'["\']?{re.escape(key)}["\']?\s*:', flags=re.IGNORECASE)
        for match in key_pattern.finditer(cleaned):
            array_start = cleaned.find("[", match.end())
            if array_start == -1:
                continue
            array_candidate = _extract_balanced_segment(cleaned, array_start, "[", "]")
            if array_candidate:
                candidates.append(f'{{"{key}": {array_candidate}}}')
    return candidates


def _loads_json_like(candidate: str) -> Any:
    candidate = candidate.strip()
    try:
        return json.loads(candidate)
    except Exception:
        return ast.literal_eval(candidate)


def _value_for_keys(parsed: Any, expected_keys: Tuple[str, ...]) -> Any:
    if isinstance(parsed, list):
        return parsed
    if not isinstance(parsed, dict):
        return None

    lower_to_key = {str(key).lower(): key for key in parsed.keys()}
    for key in expected_keys:
        if key in parsed:
            return parsed[key]
        original_key = lower_to_key.get(key.lower())
        if original_key is not None:
            return parsed[original_key]
    return None


def _extract_openie_value(real_response: str, expected_keys: Tuple[str, ...]) -> Any:
    candidates = _iter_json_object_candidates(real_response)
    candidates.extend(_iter_keyed_array_candidates(real_response, expected_keys))

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            parsed = _loads_json_like(candidate)
        except Exception:
            continue
        value = _value_for_keys(parsed, expected_keys)
        if value is not None:
            return value

    logger.debug(f"OpenIE parse failed for keys={expected_keys}. Response prefix: {real_response[:200]!r}")
    return []


def _normalize_entities(entities: Any) -> List[str]:
    if isinstance(entities, str):
        entity = entities.strip()
        return [entity] if entity else []
    if not isinstance(entities, list):
        return []

    normalized_entities = []
    for entity in entities:
        if entity is None:
            continue
        entity = str(entity).strip()
        if entity:
            normalized_entities.append(entity)
    return normalized_entities


def _normalize_triples(triples: Any) -> List[List[str]]:
    if not isinstance(triples, list):
        return []

    normalized_triples = []
    for triple in triples:
        if not isinstance(triple, (list, tuple)) or len(triple) != 3:
            continue
        if any(item is None for item in triple):
            continue
        normalized_triple = [str(item).strip() for item in triple]
        if all(normalized_triple):
            normalized_triples.append(normalized_triple)
    return normalized_triples


class ChunkInfo(TypedDict):
    num_tokens: int
    content: str
    chunk_order: List[Tuple]
    full_doc_ids: List[str]


@dataclass
class LLMInput:
    chunk_id: str
    input_message: List[Dict]


def _extract_ner_from_response(real_response):
    entities = _extract_openie_value(real_response, ("named_entities", "entities", "entity"))
    return _normalize_entities(entities)


def _extract_triples_from_response(real_response):
    triples = _extract_openie_value(real_response, ("triples", "triple", "facts", "fact"))
    return _normalize_triples(triples)


class OpenIE:
    def __init__(self, llm_model: CacheOpenAI):
        # Init prompt template manager
        self.prompt_template_manager = PromptTemplateManager(role_mapping={"system": "system", "user": "user", "assistant": "assistant"})
        self.llm_model = llm_model
        global_config = getattr(llm_model, "global_config", None)
        self.ner_max_tokens = int(getattr(global_config, "openie_ner_max_tokens", 512) or 512)
        self.triple_max_tokens = int(getattr(global_config, "openie_triple_max_tokens", 2048) or 2048)
        self.openie_num_workers = int(getattr(global_config, "openie_num_workers", 48) or 48)
        self.retry_low_quality = bool(getattr(global_config, "openie_retry_low_quality", True))
        self.retry_min_entities = int(getattr(global_config, "openie_retry_min_entities", 4) or 4)
        self.retry_min_triples = int(getattr(global_config, "openie_retry_min_triples", 2) or 2)
        self.retry_max_tokens = int(getattr(global_config, "openie_retry_max_tokens", 2048) or 2048)

    def ner(self, chunk_key: str, passage: str) -> NerRawOutput:
        # PREPROCESSING
        ner_input_message = self.prompt_template_manager.render(name='ner', passage=passage)
        raw_response = ""
        metadata = {}
        try:
            infer_kwargs = {}
            if self.ner_max_tokens is not None:
                infer_kwargs["max_completion_tokens"] = self.ner_max_tokens
            # LLM INFERENCE
            raw_response, metadata, cache_hit = self.llm_model.infer(
                messages=ner_input_message,
                **infer_kwargs,
            )
            metadata['cache_hit'] = cache_hit
            metadata['max_completion_tokens'] = self.ner_max_tokens
            if metadata['finish_reason'] == 'length':
                real_response = fix_broken_generated_json(raw_response)
            else:
                real_response = raw_response
            extracted_entities = _extract_ner_from_response(real_response)
            unique_entities = list(dict.fromkeys(extracted_entities))

        except Exception as e:
            # For any other unexpected exceptions, log them and return with the error message
            logger.warning(e)
            metadata.update({'error': str(e)})
            return NerRawOutput(
                chunk_id=chunk_key,
                response=raw_response,  # Store the error message in metadata
                unique_entities=[],
                metadata=metadata  # Store the error message in metadata
            )

        return NerRawOutput(
            chunk_id=chunk_key,
            response=raw_response,
            unique_entities=unique_entities,
            metadata=metadata
        )

    def triple_extraction(
        self,
        chunk_key: str,
        passage: str,
        named_entities: List[str],
        retry: bool = False,
        max_completion_tokens: int = None,
    ) -> TripleRawOutput:
        # PREPROCESSING
        messages = self.prompt_template_manager.render(
            name='triple_extraction',
            passage=passage,
            named_entity_json=json.dumps({"named_entities": named_entities})
        )
        if retry:
            messages = [dict(message) for message in messages]
            messages[-1]["content"] = (
                f"{messages[-1]['content']}\n\n"
                "Retry instruction: the previous extraction was sparse or invalid. "
                "Re-extract an exhaustive but faithful RDF graph from this paragraph. "
                "Include all explicit factual relations, especially aliases, dates, locations, "
                "memberships, occupations, affiliations, works, events, and descriptive object phrases. "
                "Keep every triple grounded in the paragraph and return only a JSON object with key \"triples\"."
            )

        raw_response = ""
        metadata = {}
        try:
            infer_kwargs = {}
            effective_max_tokens = max_completion_tokens if max_completion_tokens is not None else self.triple_max_tokens
            if effective_max_tokens is not None:
                infer_kwargs["max_completion_tokens"] = effective_max_tokens
            # LLM INFERENCE
            raw_response, metadata, cache_hit = self.llm_model.infer(
                messages=messages,
                **infer_kwargs,
            )
            metadata['cache_hit'] = cache_hit
            metadata['retry'] = retry
            metadata['max_completion_tokens'] = effective_max_tokens
            if metadata['finish_reason'] == 'length':
                real_response = fix_broken_generated_json(raw_response)
            else:
                real_response = raw_response
            extracted_triples = _extract_triples_from_response(real_response)
            triplets = filter_invalid_triples(triples=extracted_triples)

        except Exception as e:
            logger.warning(f"Exception for chunk {chunk_key}: {e}")
            metadata.update({'error': str(e)})
            return TripleRawOutput(
                chunk_id=chunk_key,
                response=raw_response,
                metadata=metadata,
                triples=[]
            )

        # Success
        return TripleRawOutput(
            chunk_id=chunk_key,
            response=raw_response,
            metadata=metadata,
            triples=triplets
        )

    def openie(self, chunk_key: str, passage: str) -> Dict[str, Any]:
        ner_output = self.ner(chunk_key=chunk_key, passage=passage)
        triple_output = self.triple_extraction(chunk_key=chunk_key, passage=passage, named_entities=ner_output.unique_entities)
        return {"ner": ner_output, "triplets": triple_output}

    def _low_quality_reason(self, ner_result: NerRawOutput, triple_result: TripleRawOutput) -> str:
        if triple_result.metadata.get('error'):
            return 'error'
        if triple_result.metadata.get('finish_reason') == 'length':
            return 'length'

        num_entities = len(ner_result.unique_entities)
        num_triples = len(triple_result.triples)
        if num_triples == 0:
            return 'empty_triples'
        if num_entities >= self.retry_min_entities and num_triples < self.retry_min_triples:
            return 'sparse_triples'
        return ''

    @staticmethod
    def _merge_triple_outputs(original: TripleRawOutput, retry_result: TripleRawOutput) -> TripleRawOutput:
        seen = set()
        merged_triples = []
        for triple in original.triples + retry_result.triples:
            key = tuple(triple)
            if key in seen:
                continue
            seen.add(key)
            merged_triples.append(triple)

        metadata = dict(original.metadata)
        metadata['low_quality_retry'] = {
            'base_triples': len(original.triples),
            'retry_triples': len(retry_result.triples),
            'merged_triples': len(merged_triples),
            'added_triples': max(0, len(merged_triples) - len(original.triples)),
            'retry_error': retry_result.metadata.get('error'),
            'retry_finish_reason': retry_result.metadata.get('finish_reason'),
        }

        response = retry_result.response if len(merged_triples) > len(original.triples) else original.response
        return TripleRawOutput(
            chunk_id=original.chunk_id,
            response=response,
            metadata=metadata,
            triples=merged_triples,
        )

    def batch_openie(self, chunks: Dict[str, ChunkInfo]) -> Tuple[Dict[str, NerRawOutput], Dict[str, TripleRawOutput]]:
        """
        Conduct batch OpenIE synchronously using multi-threading which includes NER and triple extraction.

        Args:
            chunks (Dict[str, ChunkInfo]): chunks to be incorporated into graph. Each key is a hashed chunk 
            and the corresponding value is the chunk info to insert.

        Returns:
            Tuple[Dict[str, NerRawOutput], Dict[str, TripleRawOutput]]:
                - A dict with keys as the chunk ids and values as the NER result instances.
                - A dict with keys as the chunk ids and values as the triple extraction result instances.
        """

        # Extract passages from the provided chunks
        chunk_passages = {chunk_key: chunk["content"] for chunk_key, chunk in chunks.items()}

        ner_results_list = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        num_cache_hit = 0

        logger.info(
            f"OpenIE max completion tokens: ner={self.ner_max_tokens}, "
            f"triple={self.triple_max_tokens}, retry={self.retry_max_tokens}"
        )
        logger.info(f"OpenIE online worker threads: {self.openie_num_workers}")

        with ThreadPoolExecutor(max_workers=self.openie_num_workers) as executor:
            # Create NER futures for each chunk
            ner_futures = {
                executor.submit(self.ner, chunk_key, passage): chunk_key
                for chunk_key, passage in chunk_passages.items()
            }

            pbar = tqdm(as_completed(ner_futures), total=len(ner_futures), desc="NER")
            for future in pbar:
                result = future.result()
                ner_results_list.append(result)
                # Update metrics based on the metadata from the result
                metadata = result.metadata
                total_prompt_tokens += metadata.get('prompt_tokens', 0)
                total_completion_tokens += metadata.get('completion_tokens', 0)
                if metadata.get('cache_hit'):
                    num_cache_hit += 1

                pbar.set_postfix({
                    'total_prompt_tokens': total_prompt_tokens,
                    'total_completion_tokens': total_completion_tokens,
                    'num_cache_hit': num_cache_hit
                })

        triple_results_list = []
        total_prompt_tokens, total_completion_tokens, num_cache_hit = 0, 0, 0
        with ThreadPoolExecutor(max_workers=self.openie_num_workers) as executor:
            # Create triple extraction futures for each chunk
            re_futures = {
                executor.submit(self.triple_extraction, ner_result.chunk_id,
                                chunk_passages[ner_result.chunk_id],
                                ner_result.unique_entities): ner_result.chunk_id
                for ner_result in ner_results_list
            }
            # Collect triple extraction results with progress bar
            pbar = tqdm(as_completed(re_futures), total=len(re_futures), desc="Extracting triples")
            for future in pbar:
                result = future.result()
                triple_results_list.append(result)
                metadata = result.metadata
                total_prompt_tokens += metadata.get('prompt_tokens', 0)
                total_completion_tokens += metadata.get('completion_tokens', 0)
                if metadata.get('cache_hit'):
                    num_cache_hit += 1
                pbar.set_postfix({
                    'total_prompt_tokens': total_prompt_tokens,
                    'total_completion_tokens': total_completion_tokens,
                    'num_cache_hit': num_cache_hit
                })

        triple_results_dict = {res.chunk_id: res for res in triple_results_list}

        if self.retry_low_quality:
            retry_items = []
            reason_counts = {}
            for ner_result in ner_results_list:
                triple_result = triple_results_dict[ner_result.chunk_id]
                reason = self._low_quality_reason(ner_result, triple_result)
                if not reason:
                    continue
                retry_items.append(ner_result)
                reason_counts[reason] = reason_counts.get(reason, 0) + 1

            if retry_items:
                logger.info(
                    f"Retrying OpenIE triple extraction for {len(retry_items)} low-quality chunks. "
                    f"reasons={reason_counts}, retry_max_tokens={self.retry_max_tokens}"
                )
                improved_chunks = 0
                added_triples = 0
                total_prompt_tokens, total_completion_tokens, num_cache_hit = 0, 0, 0
                with ThreadPoolExecutor(max_workers=self.openie_num_workers) as executor:
                    retry_futures = {
                        executor.submit(
                            self.triple_extraction,
                            ner_result.chunk_id,
                            chunk_passages[ner_result.chunk_id],
                            ner_result.unique_entities,
                            True,
                            self.retry_max_tokens,
                        ): ner_result.chunk_id
                        for ner_result in retry_items
                    }
                    pbar = tqdm(as_completed(retry_futures), total=len(retry_futures), desc="Retrying triples")
                    for future in pbar:
                        retry_result = future.result()
                        original_result = triple_results_dict[retry_result.chunk_id]
                        merged_result = self._merge_triple_outputs(original_result, retry_result)
                        delta = len(merged_result.triples) - len(original_result.triples)
                        if delta > 0:
                            improved_chunks += 1
                            added_triples += delta
                        triple_results_dict[retry_result.chunk_id] = merged_result

                        metadata = retry_result.metadata
                        total_prompt_tokens += metadata.get('prompt_tokens', 0)
                        total_completion_tokens += metadata.get('completion_tokens', 0)
                        if metadata.get('cache_hit'):
                            num_cache_hit += 1
                        pbar.set_postfix({
                            'improved_chunks': improved_chunks,
                            'added_triples': added_triples,
                            'total_prompt_tokens': total_prompt_tokens,
                            'total_completion_tokens': total_completion_tokens,
                            'num_cache_hit': num_cache_hit
                        })

                logger.info(
                    f"OpenIE low-quality retry completed: retried={len(retry_items)}, "
                    f"improved_chunks={improved_chunks}, added_triples={added_triples}"
                )

        ner_results_dict = {res.chunk_id: res for res in ner_results_list}

        return ner_results_dict, triple_results_dict
