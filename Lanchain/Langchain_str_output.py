# from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.output_parsers import YamlOutputParser
# # from langchain_core.output_parsers import JsonOutputParser

# load_dotenv()

# llm = ChatOpenAI(
#     model="gpt-5.5"
# )

# prompt = PromptTemplate(
#     template="Explain {topic}",
#     input_variables=["ML"]
# )

# chain = prompt | llm | StrOutputParser()


# response = chain.invoke(
#     {
#         "topic": "Machine Learning"
#     }
# )

# print(response)

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import YamlOutputParser

load_dotenv()

llm = ChatOpenAI(model="gpt-5.5")

parser = YamlOutputParser()

prompt = PromptTemplate(
    template="""
    Generate interview questions for {topic}.

    {{
        "topic": "{topic}",
        "difficulty": "Beginner",
        "questions": [
            "question1",
            "question2"
        ]
    }}
    """,
    input_variables=["topic"]
)

chain = prompt | llm | parser

response = chain.invoke({
    "topic": "LangChain"
})

print(response)
print(type(response))