import streamlit as st
from retriever import retrieve
from rag_pipeline import generate_answer

st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("RAG Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Ask something...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    contexts = retrieve(query)
    answer = generate_answer(query, contexts)

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
