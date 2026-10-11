import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.title("🤖 AI Text Summarizer")

text = st.text_area(
    "Enter your text",
    height=250
)

if st.button("Summarize"):

    if not text:
        st.warning("Please enter some text.")
    else:

        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0
        )

        prompt = ChatPromptTemplate.from_template(
            """
            Summarize the following text in simple language.

            Keep the summary concise and provide
            3-5 important points.

            Text:
            {text}
            """
        )

        chain = prompt | llm

        response = chain.invoke({
            "text": text
        })

        st.subheader("Summary")

        st.write(response.content)