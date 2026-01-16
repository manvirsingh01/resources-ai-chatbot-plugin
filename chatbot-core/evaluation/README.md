# LLM Evaluation Pipeline

This module provides an automated evaluation pipeline for the Jenkins chatbot using "LLM-as-a-judge" methodology.

## Overview

The evaluation pipeline assesses chatbot responses on three key metrics:

| Metric | Description | Threshold |
|--------|-------------|-----------|
| **Faithfulness** | Did the model make things up? (hallucination detection) | ≥ 0.85 |
| **Context Recall** | Did the retriever find the right documents? | ≥ 0.70 |
| **Answer Relevancy** | Did the response actually answer the question? | ≥ 0.75 |

## Quick Start

### Run Evaluation Locally (Mock Mode)

```bash
cd chatbot-core
EVAL_MOCK_MODE=true pytest evaluation/runner.py -v
```

### Run Evaluation with Real LLM Judge

```bash
cd chatbot-core
export OPENAI_API_KEY=your-api-key
pytest evaluation/runner.py -v
```

## Directory Structure

```
evaluation/
├── __init__.py
├── config.py           # Thresholds and configuration
├── metrics.py          # DeepEval metrics wrapper
├── runner.py           # Pytest test suite
├── data/
│   └── golden_dataset.json  # 100+ verified Q&A pairs
└── README.md
```

## Golden Dataset

The dataset contains 105 verified Q&A pairs across three categories:

- **Jenkins Core** (30 questions): Installation, pipelines, agents, configuration
- **Plugins** (40 questions): Git, GitHub, Slack, Docker, Kubernetes, etc.
- **Errors** (35 questions): Common error messages and troubleshooting

### Dataset Schema

```json
{
  "id": "core_001",
  "category": "jenkins_core",
  "question": "How do I install Jenkins on Ubuntu?",
  "expected_answer": "To install Jenkins on Ubuntu...",
  "ground_truth_context": "Jenkins installation requires...",
  "keywords": ["install", "ubuntu", "setup"]
}
```

### Adding New Questions

1. Edit `data/golden_dataset.json`
2. Add new question following the schema
3. Ensure category counts meet minimum requirements:
   - jenkins_core: 30+
   - plugins: 40+
   - errors: 30+

## CI Integration

The evaluation runs automatically when:
- A PR is labeled with `run-eval`
- Someone comments `/run-eval` on a PR

### Triggering Evaluation

1. Open your PR
2. Add the label `run-eval` OR comment `/run-eval`
3. Check the Actions tab for results
4. Review the evaluation report in PR comments

## Interpreting Results

### Passing Build
All three metrics exceed their thresholds. Your changes don't negatively impact response quality.

### Failing Build

| Failed Metric | Likely Cause | Action |
|--------------|--------------|--------|
| Faithfulness < 0.85 | Model generating unsupported claims | Review prompt engineering, add guardrails |
| Context Recall < 0.70 | Retriever not finding relevant docs | Check retrieval logic, embedding quality |
| Answer Relevancy < 0.75 | Responses not addressing questions | Review prompt, check for topic drift |

### Sample Report

```
## 📊 LLM Evaluation Report

### Scores
| Metric | Score | Threshold | Status |
|--------|-------|-----------|--------|
| Faithfulness | 0.892 | 0.85 | ✅ Pass |
| Context Recall | 0.756 | 0.70 | ✅ Pass |
| Answer Relevancy | 0.823 | 0.75 | ✅ Pass |

### Overall Status: ✅ PASSED
```

## Configuration

Edit `config.py` to adjust:

```python
THRESHOLDS = {
    "faithfulness": 0.85,
    "context_recall": 0.70,
    "answer_relevancy": 0.75,
}

SAMPLE_SIZE = 20  # Questions per run
EVAL_MODEL = "gpt-4o-mini"  # Judge LLM
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | API key for judge LLM | Required |
| `EVAL_MOCK_MODE` | Skip real LLM calls | `false` |
| `EVAL_RANDOM_SEED` | Reproducible sampling | `42` |

## Troubleshooting

### "DeepEval not installed"
```bash
pip install deepeval
```

### "Golden dataset not found"
Ensure you're running from `chatbot-core` directory.

### API costs too high
Reduce `SAMPLE_SIZE` in config.py or use mock mode for development.
