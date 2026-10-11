from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric
)

from deepeval.test_case import LLMTestCase


def evaluate_hr_answer(
    question,
    actual_answer,
    retrieval_context
):

    # =========================================================
    # Create DeepEval test case
    # =========================================================

    test_case = LLMTestCase(
        input=question,
        actual_output=actual_answer,
        retrieval_context=retrieval_context
    )

    # =========================================================
    # Faithfulness
    # =========================================================

    faithfulness = FaithfulnessMetric(
        threshold=0.7,
        model="gpt-4.1-mini",
        include_reason=True
    )

    # =========================================================
    # Relevance
    # =========================================================

    relevance = AnswerRelevancyMetric(
        threshold=0.7,
        model="gpt-4.1-mini",
        include_reason=True
    )

    # =========================================================
    # Measure each metric directly
    # =========================================================

    print("\nEvaluating Faithfulness...")

    faithfulness.measure(test_case)

    print("\nEvaluating Relevance...")

    relevance.measure(test_case)

    # =========================================================
    # Get scores
    # =========================================================

    faithfulness_score = faithfulness.score
    faithfulness_reason = faithfulness.reason

    relevance_score = relevance.score
    relevance_reason = relevance.reason

    # =========================================================
    # Return
    # =========================================================

    return {
        "faithfulness_score": faithfulness_score,
        "faithfulness_reason": faithfulness_reason,

        "relevance_score": relevance_score,
        "relevance_reason": relevance_reason
    }