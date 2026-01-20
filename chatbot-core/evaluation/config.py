"""
Configuration for LLM evaluation pipeline.
"""

# Evaluation thresholds - tests will fail if scores fall below these values
THRESHOLDS = {
    "faithfulness": 0.85,      # Did it make things up?
    "context_recall": 0.70,    # Did it find the right document?
    "answer_relevancy": 0.75,  # Did it answer the question?
}

# Judge LLM model configuration
EVAL_MODEL = "gpt-4o-mini"

# Number of questions to sample per evaluation run (to save API costs)
SAMPLE_SIZE = 20

# Path to golden dataset
GOLDEN_DATASET_PATH = "evaluation/data/golden_dataset.json"

# Minimum number of questions required in dataset
MIN_DATASET_SIZE = 100

# Categories and their minimum counts
CATEGORY_REQUIREMENTS = {
    "jenkins_core": 30,
    "plugins": 40,
    "errors": 30,
}
