import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
    
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
    temperature=0
)

prompt = ChatPromptTemplate.from_template(
    """You are an HR policy assistant.

Answer the employee's question ONLY using the HR context.

Do not invent HR policies.

If the answer cannot be found in the context, say:
"I could not find this information in the HR document."

HR CONTEXT:
{context}

EMPLOYEE QUESTION:
{question}

Give a clear and concise answer."""
)

def ask_hr_agent(question: str):
    documents = retriever.invoke(question)
    retrieval_context = [doc.page_content for doc in documents]
    context = "\n\n".join(retrieval_context)

    messages = prompt.format_messages(
        context=context,
        question=question
    )
    response = llm.invoke(messages)

    return {
        "question": question,
        "answer": response.content,
        "retrieval_context": retrieval_context
    }
