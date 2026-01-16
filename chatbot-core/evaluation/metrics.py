"""
Evaluation metrics wrapper for LLM-as-a-judge evaluation.

This module provides a unified interface for running DeepEval metrics
on chatbot responses to assess quality.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass

# Check if deepeval is available
try:
    from deepeval import evaluate
    from deepeval.metrics import (
        FaithfulnessMetric,
        ContextualRecallMetric,
        AnswerRelevancyMetric,
    )
    from deepeval.test_case import LLMTestCase
    DEEPEVAL_AVAILABLE = True
except ImportError:
    DEEPEVAL_AVAILABLE = False

from evaluation.config import EVAL_MODEL, THRESHOLDS


@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    faithfulness: float
    context_recall: float
    answer_relevancy: float
    passed: bool
    details: Dict[str, Any]


class EvaluationMetrics:
    """
    Wrapper class for DeepEval metrics used in LLM-as-a-judge evaluation.
    
    Metrics:
    - Faithfulness: Did the response make things up? (hallucination detection)
    - Context Recall: Did the retriever find the right documents?
    - Answer Relevancy: Did the response actually answer the question?
    """
    
    def __init__(self, model: Optional[str] = None, mock_mode: bool = False):
        """
        Initialize evaluation metrics.
        
        Args:
            model: The LLM model to use as judge (default: gpt-4o-mini)
            mock_mode: If True, return mock scores for testing without API calls
        """
        self.model = model or EVAL_MODEL
        self.mock_mode = mock_mode
        self.thresholds = THRESHOLDS
        
        if not mock_mode and not DEEPEVAL_AVAILABLE:
            raise ImportError(
                "DeepEval is required for evaluation. "
                "Install it with: pip install deepeval"
            )
        
        if not mock_mode:
            self._init_metrics()
    
    def _init_metrics(self):
        """Initialize DeepEval metric instances."""
        self.faithfulness_metric = FaithfulnessMetric(
            threshold=self.thresholds["faithfulness"],
            model=self.model,
            include_reason=True
        )
        self.context_recall_metric = ContextualRecallMetric(
            threshold=self.thresholds["context_recall"],
            model=self.model,
            include_reason=True
        )
        self.answer_relevancy_metric = AnswerRelevancyMetric(
            threshold=self.thresholds["answer_relevancy"],
            model=self.model,
            include_reason=True
        )
    
    def evaluate_single(
        self,
        question: str,
        answer: str,
        context: str,
        expected_answer: str
    ) -> EvaluationResult:
        """
        Evaluate a single Q&A pair against all metrics.
        
        Args:
            question: The user's question
            answer: The chatbot's generated answer
            context: The retrieved context used to generate the answer
            expected_answer: The expected/ground truth answer
        
        Returns:
            EvaluationResult with scores for all metrics
        """
        if self.mock_mode:
            return self._mock_evaluate()
        
        # Create test case
        test_case = LLMTestCase(
            input=question,
            actual_output=answer,
            expected_output=expected_answer,
            retrieval_context=[context] if isinstance(context, str) else context
        )
        
        # Run each metric
        self.faithfulness_metric.measure(test_case)
        self.context_recall_metric.measure(test_case)
        self.answer_relevancy_metric.measure(test_case)
        
        # Collect results
        faithfulness_score = self.faithfulness_metric.score
        context_recall_score = self.context_recall_metric.score
        answer_relevancy_score = self.answer_relevancy_metric.score
        
        # Check if all thresholds are met
        passed = (
            faithfulness_score >= self.thresholds["faithfulness"] and
            context_recall_score >= self.thresholds["context_recall"] and
            answer_relevancy_score >= self.thresholds["answer_relevancy"]
        )
        
        return EvaluationResult(
            faithfulness=faithfulness_score,
            context_recall=context_recall_score,
            answer_relevancy=answer_relevancy_score,
            passed=passed,
            details={
                "faithfulness_reason": self.faithfulness_metric.reason,
                "context_recall_reason": self.context_recall_metric.reason,
                "answer_relevancy_reason": self.answer_relevancy_metric.reason,
            }
        )
    
    def _mock_evaluate(self) -> EvaluationResult:
        """Return mock evaluation results for testing."""
        import random
        
        # Generate realistic mock scores
        faithfulness = random.uniform(0.80, 0.95)
        context_recall = random.uniform(0.65, 0.85)
        answer_relevancy = random.uniform(0.70, 0.90)
        
        passed = (
            faithfulness >= self.thresholds["faithfulness"] and
            context_recall >= self.thresholds["context_recall"] and
            answer_relevancy >= self.thresholds["answer_relevancy"]
        )
        
        return EvaluationResult(
            faithfulness=faithfulness,
            context_recall=context_recall,
            answer_relevancy=answer_relevancy,
            passed=passed,
            details={
                "faithfulness_reason": "Mock evaluation - no API call made",
                "context_recall_reason": "Mock evaluation - no API call made",
                "answer_relevancy_reason": "Mock evaluation - no API call made",
            }
        )
    
    def evaluate_batch(
        self,
        test_cases: list
    ) -> Dict[str, Any]:
        """
        Evaluate a batch of test cases and return aggregate scores.
        
        Args:
            test_cases: List of dicts with keys: question, answer, context, expected_answer
        
        Returns:
            Dict with average scores and individual results
        """
        results = []
        for tc in test_cases:
            result = self.evaluate_single(
                question=tc["question"],
                answer=tc["answer"],
                context=tc["context"],
                expected_answer=tc["expected_answer"]
            )
            results.append(result)
        
        # Calculate averages
        avg_faithfulness = sum(r.faithfulness for r in results) / len(results)
        avg_context_recall = sum(r.context_recall for r in results) / len(results)
        avg_answer_relevancy = sum(r.answer_relevancy for r in results) / len(results)
        
        # Check overall pass
        overall_passed = (
            avg_faithfulness >= self.thresholds["faithfulness"] and
            avg_context_recall >= self.thresholds["context_recall"] and
            avg_answer_relevancy >= self.thresholds["answer_relevancy"]
        )
        
        return {
            "average_scores": {
                "faithfulness": avg_faithfulness,
                "context_recall": avg_context_recall,
                "answer_relevancy": avg_answer_relevancy,
            },
            "thresholds": self.thresholds,
            "overall_passed": overall_passed,
            "total_cases": len(results),
            "passed_cases": sum(1 for r in results if r.passed),
            "individual_results": results,
        }
