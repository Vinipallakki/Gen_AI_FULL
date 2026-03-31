# import os
# from openai import OpenAI
# from dotenv import load_dotenv

# load_dotenv()
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# def get_prompt(role, user_input):
#     if role == "teacher":
#         return f"You are a friendly teacher. Explain clearly with examples:\n{user_input}"
#     elif role == "quiz":
#         return f"You are a quiz generator. Create 3 MCQ questions with 4 options:\n{user_input}"
#     elif role == "examiner":
#         return f"You are an examiner. Evaluate this answer and give score out of 10 with feedback:\n{user_input}"
#     elif role == "summarizer":
#         return f"Summarize this in simple bullet points:\n{user_input}"
#     elif role == "motivator":
#         return f"Give short motivational advice for studying:\n{user_input}"
#     return user_input


# def ask_ai(prompt):
#     response = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[
#             {"role": "system", "content": "You are a smart study assistant."},
#             {"role": "user", "content": prompt}
#         ]
#     )
#     return response.choices[0].message.content

# def main():
#     print("📚 Smart Study Assistant")
#     while True:
#         print("\nChoose Role:")
#         print("1. Teacher")
#         print("2. Quiz")
#         print("3. Examiner")
#         print("4. Summarizer")
#         print("5. Motivator")
#         print("6. Exit")

#         choice = input("Enter choice: ")
#         if choice == "6":
#             break

#         role_map = {
#             "1": "teacher",
#             "2": "quiz",
#             "3": "examiner",
#             "4": "summarizer",
#             "5": "motivator"
#         }

#         role = role_map.get(choice)
#         user_input = input("\nEnter your topic or answer: ")
#         prompt = get_prompt(role, user_input)
#         response = ask_ai(prompt)
#         print("\n🤖 Response:\n", response)

# if __name__ == "__main__":
#     main()


def get_prompt(role, user_input):
    if role == "teacher":
        return f"You are a friendly teacher. Explain clearly with examples:\n{user_input}"
    elif role == "quiz":
        return f"You are a quiz generator. Create 3 MCQ questions with 4 options:\n{user_input}"
    elif role == "examiner":
        return f"You are an examiner. Evaluate this answer and give score out of 10 with feedback:\n{user_input}"
    elif role == "summarizer":
        return f"Summarize this in simple bullet points:\n{user_input}"
    elif role == "motivator":
        return f"Give short motivational advice for studying:\n{user_input}"
    return user_input


def ask_ai(prompt):
    response = ollama.chat(
        model="llama3",   # you can change to mistral, phi, etc.
        messages=[
            {"role": "system", "content": "You are a smart study assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response["message"]["content"]


def main():
    print("📚 Smart Study Assistant (Ollama)")
    while True:
        print("\nChoose Role:")
        print("1. Teacher")
        print("2. Quiz")
        print("3. Examiner")
        print("4. Summarizer")
        print("5. Motivator")
        print("6. Exit")

        choice = input("Enter choice: ")
        if choice == "6":
            break

        role_map = {
            "1": "teacher",
            "2": "quiz",
            "3": "examiner",
            "4": "summarizer",
            "5": "motivator"
        }

        role = role_map.get(choice)
        user_input = input("\nEnter your topic or answer: ")
        prompt = get_prompt(role, user_input)

        response = ask_ai(prompt)
        print("\n🤖 Response:\n", response)


if __name__ == "__main__":
    main()