# import os
# from openai import OpenAI

# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# def generate_answer(query, contexts):
#     context_text = "\n".join(contexts)
#     prompt = f"""
# Answer based only on the context below.

# Context:
# {context_text}

# Question:
# {query}
# """
#     response = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[{"role": "user", "content": prompt}]
#     )
#     return response.choices[0].message.content

# rag_pipeline.py

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_answer(query, contexts):

    context_text = "\n".join(contexts)

    # 🔥 Inject company info here
    company_info = """
    Company Name: ABC Corp

    HR Assistant Name: Ava

    HR Responsibilities:
    - Employee onboarding
    - Leave management
    - Payroll support
    - Employee relations
    """

    system_prompt = f"""
    You are a professional HR assistant.

    About yourself:
    - Your name is Ava
    - You work in HR department at ABC Corp

    Instructions:
    1. Use CONTEXT for company-specific answers
    2. Use COMPANY INFO for general HR identity
    3. If answer not found, say "I don't have that information"
    4. Be polite and professional

    COMPANY INFO:
    {company_info}
    """

    user_prompt = f"""
    Context:
    {context_text}

    Question:
    {query}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    return response.choices[0].message.content