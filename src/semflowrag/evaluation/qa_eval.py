from typing import List, Dict, Tuple, Optional, Union, Callable
from collections import Counter
import numpy as np

from .base import BaseMetric
from ..utils.logging_utils import get_logger
from ..utils.config_utils import BaseConfig
from ..utils.eval_utils import normalize_answer

logger = get_logger(__name__)

# Reference: MRQA official eval
class QAExactMatch(BaseMetric):
    metric_name: str = "qa_exact_match"

    def __init__(self, global_config: Optional[BaseConfig] = None):
        super().__init__(global_config)

    def calculate_metric_scores(self, gold_answers: List[List[str]], predicted_answers: List[str], aggregation_fn: Callable = np.max) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        """
        Calculates the Exact Match (EM) score.

        Args:
            gold_answers (List[List[str]]): List of lists containing ground truth answers.
            predicted_answers (List[str]): List of predicted answers.
            aggregation_fn (Callable): Function to aggregate scores across multiple gold answers (default: np.max).

        Returns:
            Tuple[Dict[str, float], List[Dict[str, float]]]: 
                - A dictionary with the averaged EM score.
                - A list of dictionaries with EM scores for each example.
        """
        assert len(gold_answers) == len(predicted_answers), "Length of gold answers and predicted answers should be the same."

        example_eval_results = []
        total_em = 0

        for gold_list, predicted in zip(gold_answers, predicted_answers):
            em_scores = [1.0 if normalize_answer(gold) == normalize_answer(predicted) else 0.0 for gold in gold_list]
            aggregated_em = aggregation_fn(em_scores)
            example_eval_results.append({"ExactMatch": aggregated_em})
            total_em += aggregated_em

        avg_em = total_em / len(gold_answers) if gold_answers else 0.0
        pooled_eval_results = {"ExactMatch": avg_em}

        return pooled_eval_results, example_eval_results

class QAF1Score(BaseMetric):
    metric_name: str = "qa_f1_score"

    def __init__(self, global_config: Optional[BaseConfig] = None):
        super().__init__(global_config)

    def calculate_metric_scores(self, gold_answers: List[List[str]], predicted_answers: List[str], aggregation_fn: Callable = np.max) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        """
        Calculates token-level precision, recall, and F1 score.

        Args:
            gold_answers (List[List[str]]): List of lists containing ground truth answers.
            predicted_answers (List[str]): List of predicted answers.
            aggregation_fn (Callable): Function to aggregate scores across multiple gold answers (default: np.max).

        Returns:
            Tuple[Dict[str, float], List[Dict[str, float]]]: 
                - A dictionary with the averaged token-level QA metrics.
                - A list of dictionaries with token-level QA metrics for each example.
        """
        assert len(gold_answers) == len(predicted_answers), "Length of gold answers and predicted answers should be the same."

        def compute_prf(gold: str, predicted: str) -> Tuple[float, float, float]:
            gold_tokens = normalize_answer(gold).split()
            predicted_tokens = normalize_answer(predicted).split()
            common = Counter(predicted_tokens) & Counter(gold_tokens)
            num_same = sum(common.values())

            if num_same == 0 or len(gold_tokens) == 0 or len(predicted_tokens) == 0:
                return 0.0, 0.0, 0.0

            precision = 1.0 * num_same / len(predicted_tokens)
            recall = 1.0 * num_same / len(gold_tokens)
            f1 = 2 * (precision * recall) / (precision + recall)
            return precision, recall, f1

        example_eval_results = []
        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0

        for gold_list, predicted in zip(gold_answers, predicted_answers):
            prf_scores = [compute_prf(gold, predicted) for gold in gold_list]
            if prf_scores:
                # Report precision/recall from the same gold answer that gives
                # the best F1, rather than mixing metrics across gold answers.
                best_idx = int(np.argmax([score[2] for score in prf_scores]))
                aggregated_precision, aggregated_recall, aggregated_f1 = prf_scores[best_idx]
            else:
                aggregated_precision, aggregated_recall, aggregated_f1 = 0.0, 0.0, 0.0

            example_eval_results.append({
                "Precision": aggregated_precision,
                "Recall": aggregated_recall,
                "F1": aggregated_f1
            })
            total_precision += aggregated_precision
            total_recall += aggregated_recall
            total_f1 += aggregated_f1

        avg_precision = total_precision / len(gold_answers) if gold_answers else 0.0
        avg_recall = total_recall / len(gold_answers) if gold_answers else 0.0
        avg_f1 = total_f1 / len(gold_answers) if gold_answers else 0.0
        pooled_eval_results = {
            "Precision": avg_precision,
            "Recall": avg_recall,
            "F1": avg_f1
        }

        return pooled_eval_results, example_eval_results
