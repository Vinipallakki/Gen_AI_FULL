from agent import investigate_application, previous_issue


def main():
    print("=" * 50)
    print("OpenClaw-Style DevOps Agent Demo")
    print("=" * 50)

    while True:
        user_input = input("\nYou: ").strip().lower()

        if user_input in {"exit", "quit"}:
            print("Agent: Goodbye!")
            break

        if "previous" in user_input or "earlier" in user_input:
            print("\nAgent:")
            print(previous_issue())
        elif "check" in user_input or "status" in user_input or "application" in user_input:
            print("\nAgent:")
            print(investigate_application())
        else:
            print("Agent: Try: 'Check my payment application' or 'What was the previous issue?'")


if __name__ == "__main__":
    main()
