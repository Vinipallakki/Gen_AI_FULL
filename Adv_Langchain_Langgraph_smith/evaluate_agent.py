
import os
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client
from langsmith.evaluation import evaluate

from lang_smith import executor

load_dotenv(Path(__file__).resolve().parent / ".env")

client = Client()

DATASET_NAME = "calculator-agent-tests"

# Create the dataset if it doesn't already exist
try:
    dataset = client.read_dataset(
        dataset_name=DATASET_NAME
    )
    print("Using existing dataset:", DATASET_NAME)

except Exception:
    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="Basic calculator and word-count tests"
    )

    examples = [
        ({"question": "Calculate 10 + 20"}, {"answer": "30"}),
        ({"question": "Calculate 5 * 6"}, {"answer": "30"}),
        ({"question": "Calculate 100 / 4"}, {"answer": "25"}),
        (
            {"question": 'Count words in "I love Python"'},
            {"answer": "3"}
        ),
    ]

    for inputs, outputs in examples:
        client.create_example(
            inputs=inputs,
            outputs=outputs,
            dataset_id=dataset.id,
        )

def target(inputs: dict) -> dict:
    result = executor.invoke(
        {"input": inputs["question"]}
    )
    return {"answer": result["output"]}

def correctness_evaluator(
    inputs: dict,
    outputs: dict,
    reference_outputs: dict,
) -> dict:
    expected = str(reference_outputs["answer"]).strip()
    actual = str(outputs["answer"]).strip()

    # Basic numeric answer check for this demo
    passed = expected in actual

    return {
        "key": "answer_correctness",
        "score": int(passed),
        "comment": f"Expected {expected}; got {actual}",
    }

results = evaluate(
    target,
    data=DATASET_NAME,
    evaluators=[correctness_evaluator],
    experiment_prefix="calculator-agent",
    max_concurrency=1,  
)

print("\nEvaluation complete.")
print("Open LangSmith to inspect the experiment results.")
