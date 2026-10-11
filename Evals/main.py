from hr_agent import ask_hr_agent
from evaluate import evaluate_hr_answer


def main():

    print("=" * 70)
    print("              HR AGENT + DEEPEVAL")
    print("=" * 70)

    # =========================================================
    # 1. User Question
    # =========================================================

    question = input(
        "\nEnter HR question: "
    ).strip()

    if not question:

        print("Question cannot be empty.")

        return

    # =========================================================
    # 2. Run HR Agent
    # =========================================================

    print("\nRunning HR Agent...")

    result = ask_hr_agent(question)

    actual_answer = result["answer"]

    retrieval_context = result["retrieval_context"]

    # =========================================================
    # 3. Agent Answer
    # =========================================================

    print("\n")
    print("=" * 70)
    print("AGENT ANSWER")
    print("=" * 70)

    print(actual_answer)

    # =========================================================
    # 4. Retrieved Context
    # =========================================================

    print("\n")
    print("=" * 70)
    print("RETRIEVED HR CONTEXT")
    print("=" * 70)

    for i, context in enumerate(
        retrieval_context,
        start=1
    ):

        print(f"\n--- Context {i} ---")

        print(context)

    # =========================================================
    # 5. DeepEval
    # =========================================================

    print("\n")
    print("=" * 70)
    print("RUNNING DEEPEVAL")
    print("=" * 70)

    evaluation = evaluate_hr_answer(
        question=question,
        actual_answer=actual_answer,
        retrieval_context=retrieval_context
    )

    # =========================================================
    # 6. Read scores
    # =========================================================

    faithfulness_score = evaluation[
        "faithfulness_score"
    ]

    faithfulness_reason = evaluation[
        "faithfulness_reason"
    ]

    relevance_score = evaluation[
        "relevance_score"
    ]

    relevance_reason = evaluation[
        "relevance_reason"
    ]

    # =========================================================
    # 7. Evaluation Results
    # =========================================================

    print("\n")
    print("=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    print(
        f"\nFaithfulness / Groundedness : "
        f"{faithfulness_score:.2f}"
    )

    print(
        f"Reason: {faithfulness_reason}"
    )

    print(
        f"\nRelevance : "
        f"{relevance_score:.2f}"
    )

    print(
        f"Reason: {relevance_reason}"
    )

    # =========================================================
    # 8. Correctness
    # =========================================================

    print("\n")
    print("=" * 70)
    print("CORRECTNESS")
    print("=" * 70)

    print("\nCorrectness : N/A")

    print(
        "Correctness requires a golden/reference answer."
    )

    print(
        "It should be evaluated separately using "
        "an offline evaluation dataset."
    )

    # =========================================================
    # 9. Final Production Decision
    # =========================================================

    print("\n")
    print("=" * 70)
    print("FINAL EVALUATION")
    print("=" * 70)

    if (
        faithfulness_score >= 0.7
        and relevance_score >= 0.7
    ):

        print("\nPASS")

        print(
            "Answer is sufficiently grounded "
            "and relevant."
        )

    else:

        print("\nFAIL")

        print(
            "Answer requires investigation."
        )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()