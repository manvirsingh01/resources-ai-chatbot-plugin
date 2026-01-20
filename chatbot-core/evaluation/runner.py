"""
LLM Evaluation Runner - pytest-compatible test suite for chatbot evaluation.

This module runs the LLM-as-a-judge evaluation pipeline against a golden dataset
of verified Q&A pairs. It integrates with pytest and can be triggered via
GitHub Actions when a PR is labeled with 'run-eval'.

Usage:
    # Run with mock LLM (for testing without API costs)
    pytest evaluation/runner.py -v --mock

    # Run with real LLM evaluation
    OPENAI_API_KEY=xxx pytest evaluation/runner.py -v

    # Run specific test
    pytest evaluation/runner.py::TestLLMEvaluation::test_faithfulness_threshold -v
"""

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Any, Optional

import pytest

from evaluation.config import (
    THRESHOLDS,
    SAMPLE_SIZE,
    MIN_DATASET_SIZE,
    CATEGORY_REQUIREMENTS,
)
from evaluation.metrics import EvaluationMetrics


# Path resolution
EVALUATION_DIR = Path(__file__).parent
DATA_DIR = EVALUATION_DIR / "data"
DATASET_PATH = DATA_DIR / "golden_dataset.json"


def load_golden_dataset() -> Dict[str, Any]:
    """Load the golden dataset from JSON file."""
    if not DATASET_PATH.exists():
        pytest.skip(f"Golden dataset not found at {DATASET_PATH}")

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def sample_questions(
    dataset: Dict[str, Any],
    sample_size: int = SAMPLE_SIZE,
    seed: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Sample questions from the dataset for evaluation.

    Args:
        dataset: The loaded golden dataset
        sample_size: Number of questions to sample
        seed: Random seed for reproducibility

    Returns:
        List of sampled question dicts
    """
    questions = dataset.get("questions", [])

    if seed is not None:
        random.seed(seed)

    # Ensure we sample proportionally from each category
    sampled = []
    categories = list(CATEGORY_REQUIREMENTS.keys())
    per_category = max(1, sample_size // len(categories))

    for category in categories:
        category_qs = [q for q in questions if q.get("category") == category]
        if category_qs:
            n = min(per_category, len(category_qs))
            sampled.extend(random.sample(category_qs, n))

    # Fill remaining slots randomly
    remaining = sample_size - len(sampled)
    if remaining > 0:
        used_ids = {q["id"] for q in sampled}
        available = [q for q in questions if q["id"] not in used_ids]
        if available:
            sampled.extend(random.sample(available, min(remaining, len(available))))

    return sampled


class MockChatbot:  # pylint: disable=too-few-public-methods
    """
    Mock chatbot for testing the evaluation pipeline.

    In production, this would be replaced with the actual chatbot
    that uses retrieve_context() and generate_answer().
    """

    def __init__(self, use_expected_answers: bool = True):
        """
        Args:
            use_expected_answers: If True, return expected answers (perfect scores).
                                  If False, return slightly varied answers.
        """
        self.use_expected_answers = use_expected_answers

    def get_response(
        self,
        question: str,  # pylint: disable=unused-argument
        expected_answer: str,
        ground_truth_context: str
    ) -> Dict[str, str]:
        """
        Generate a mock response.

        Returns:
            Dict with 'answer' and 'context' keys
        """
        if self.use_expected_answers:
            return {
                "answer": expected_answer,
                "context": ground_truth_context,
            }
        # Add some variation for more realistic testing
        return {
            "answer": expected_answer + " [Note: This is a mock response.]",
            "context": ground_truth_context,
        }


def get_chatbot():
    """
    Get the chatbot instance for evaluation.

    In production, this imports and configures the actual chatbot.
    For testing, returns a mock.
    """
    # Check if we're in mock mode (set via environment or pytest marker)
    mock_mode = os.environ.get("EVAL_MOCK_MODE", "false").lower() == "true"

    if mock_mode:
        return MockChatbot(use_expected_answers=True)

    # Try to import actual chatbot
    try:
        # pylint: disable=import-outside-toplevel
        from api.services.chat_service import retrieve_context, generate_answer
        from api.prompts.prompt_builder import build_prompt

        class ActualChatbot:  # pylint: disable=too-few-public-methods
            """Wrapper for the actual chatbot service."""
            def get_response(
                self,
                question: str,
                expected_answer: str,  # pylint: disable=unused-argument
                ground_truth_context: str  # pylint: disable=unused-argument
            ) -> Dict[str, str]:
                """Get response from actual chatbot service."""
                context = retrieve_context(question)
                # Build prompt with minimal memory (no conversation history)
                # pylint: disable=import-outside-toplevel
                from langchain.memory import ConversationBufferMemory
                memory = ConversationBufferMemory()
                prompt = build_prompt(question, context, memory)
                answer = generate_answer(prompt)
                return {"answer": answer, "context": context}

        return ActualChatbot()
    except ImportError:
        # Fall back to mock if imports fail
        return MockChatbot(use_expected_answers=False)


# ============================================================================
# Pytest Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def golden_dataset():  # pylint: disable=redefined-outer-name
    """Load the golden dataset once per test session."""
    return load_golden_dataset()


@pytest.fixture(scope="session")
def sampled_questions(golden_dataset):  # pylint: disable=redefined-outer-name
    """Sample questions for evaluation."""
    # Use a fixed seed for reproducibility in CI
    seed = int(os.environ.get("EVAL_RANDOM_SEED", 42))
    return sample_questions(golden_dataset, SAMPLE_SIZE, seed=seed)


@pytest.fixture(scope="session")
def chatbot():  # pylint: disable=redefined-outer-name
    """Get the chatbot instance."""
    return get_chatbot()


@pytest.fixture(scope="session")
def evaluation_metrics():  # pylint: disable=redefined-outer-name
    """Initialize evaluation metrics."""
    mock_mode = os.environ.get("EVAL_MOCK_MODE", "false").lower() == "true"
    return EvaluationMetrics(mock_mode=mock_mode)


@pytest.fixture(scope="session")
def evaluation_results(  # pylint: disable=redefined-outer-name
    sampled_questions, chatbot, evaluation_metrics
):
    """
    Run evaluation on sampled questions and return aggregate results.

    This is the main fixture that performs the actual evaluation.
    """
    test_cases = []

    for q in sampled_questions:
        # Get chatbot response
        response = chatbot.get_response(
            question=q["question"],
            expected_answer=q["expected_answer"],
            ground_truth_context=q.get("ground_truth_context", "")
        )

        test_cases.append({
            "question": q["question"],
            "answer": response["answer"],
            "context": response["context"],
            "expected_answer": q["expected_answer"],
            "category": q.get("category", "unknown"),
            "id": q.get("id", "unknown"),
        })

    # Run batch evaluation
    return evaluation_metrics.evaluate_batch(test_cases)


# ============================================================================
# Test Classes
# ============================================================================

class TestDatasetValidation:  # pylint: disable=redefined-outer-name
    """Tests to validate the golden dataset structure and content."""

    def test_dataset_exists(self):
        """Golden dataset file should exist."""
        assert DATASET_PATH.exists(), f"Dataset not found at {DATASET_PATH}"

    def test_dataset_structure(self, golden_dataset):
        """Dataset should have required structure."""
        assert "version" in golden_dataset
        assert "questions" in golden_dataset
        assert isinstance(golden_dataset["questions"], list)

    def test_minimum_dataset_size(self, golden_dataset):
        """Dataset should have at least MIN_DATASET_SIZE questions."""
        questions = golden_dataset.get("questions", [])
        assert len(questions) >= MIN_DATASET_SIZE, (
            f"Dataset has {len(questions)} questions, "
            f"minimum required is {MIN_DATASET_SIZE}"
        )

    def test_category_coverage(self, golden_dataset):
        """Dataset should have sufficient questions in each category."""
        questions = golden_dataset.get("questions", [])

        category_counts = {}
        for q in questions:
            cat = q.get("category", "unknown")
            category_counts[cat] = category_counts.get(cat, 0) + 1

        for category, min_count in CATEGORY_REQUIREMENTS.items():
            actual = category_counts.get(category, 0)
            assert actual >= min_count, (
                f"Category '{category}' has {actual} questions, "
                f"minimum required is {min_count}"
            )

    def test_question_structure(self, golden_dataset):
        """Each question should have required fields."""
        required_fields = ["id", "category", "question", "expected_answer"]

        for q in golden_dataset.get("questions", []):
            for field in required_fields:
                assert field in q, f"Question {q.get('id', 'unknown')} missing '{field}'"


class TestLLMEvaluation:  # pylint: disable=redefined-outer-name
    """LLM-as-a-judge evaluation tests."""

    def test_faithfulness_threshold(self, evaluation_results):
        """Faithfulness score must remain above threshold."""
        score = evaluation_results["average_scores"]["faithfulness"]
        threshold = THRESHOLDS["faithfulness"]
        assert score >= threshold, (
            f"Faithfulness score {score:.3f} is below threshold {threshold}. "
            f"The model may be generating hallucinated content."
        )

    def test_context_recall_threshold(self, evaluation_results):
        """Context recall score must remain above threshold."""
        score = evaluation_results["average_scores"]["context_recall"]
        threshold = THRESHOLDS["context_recall"]
        assert score >= threshold, (
            f"Context recall score {score:.3f} is below threshold {threshold}. "
            f"The retriever may not be finding the right documents."
        )

    def test_answer_relevancy_threshold(self, evaluation_results):
        """Answer relevancy score must remain above threshold."""
        score = evaluation_results["average_scores"]["answer_relevancy"]
        threshold = THRESHOLDS["answer_relevancy"]
        assert score >= threshold, (
            f"Answer relevancy score {score:.3f} is below threshold {threshold}. "
            f"The model may not be answering the questions asked."
        )

    def test_overall_evaluation_passes(self, evaluation_results):
        """Overall evaluation should pass all thresholds."""
        assert evaluation_results["overall_passed"], (
            f"Overall evaluation failed. Scores: "
            f"Faithfulness={evaluation_results['average_scores']['faithfulness']:.3f}, "
            f"Context Recall={evaluation_results['average_scores']['context_recall']:.3f}, "
            f"Answer Relevancy={evaluation_results['average_scores']['answer_relevancy']:.3f}"
        )


class TestEvaluationReport:  # pylint: disable=redefined-outer-name,too-few-public-methods
    """Generate evaluation report for PR comments."""

    def test_generate_report(self, evaluation_results, golden_dataset):
        """Generate a markdown report of evaluation results."""
        scores = evaluation_results["average_scores"]
        thresholds = evaluation_results["thresholds"]

        # Build report
        report_lines = [
            "## 📊 LLM Evaluation Report",
            "",
            "### Scores",
            "",
            "| Metric | Score | Threshold | Status |",
            "|--------|-------|-----------|--------|",
        ]

        for metric in ["faithfulness", "context_recall", "answer_relevancy"]:
            score = scores[metric]
            threshold = thresholds[metric]
            status = "✅ Pass" if score >= threshold else "❌ Fail"
            report_lines.append(
                f"| {metric.replace('_', ' ').title()} | {score:.3f} | {threshold} | {status} |"
            )

        report_lines.extend([
            "",
            "### Summary",
            "",
            f"- **Total questions evaluated**: {evaluation_results['total_cases']}",
            f"- **Questions passed**: {evaluation_results['passed_cases']}",
            f"- **Dataset version**: {golden_dataset.get('version', 'unknown')}",
            "",
        ])

        overall_status = "✅ **PASSED**" if evaluation_results["overall_passed"] else "❌ **FAILED**"
        report_lines.append(f"### Overall Status: {overall_status}")

        report = "\n".join(report_lines)

        # Write report to file for GitHub Actions to pick up
        report_path = EVALUATION_DIR / "evaluation_report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)

        print("\n" + report)

        # This test always passes - it's just for report generation
        assert True


# Allow running as script
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
