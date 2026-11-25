
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Configure page
st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #101827, #172554);
    color: white;
}
.title {
    text-align: center;
    color: #60A5FA;
    font-size: 40px;
    font-weight: bold;
}
.subtitle {
    text-align: center;
    color: #CBD5E1;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

# Initialize OpenAI
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("Please configure OPENAI_API_KEY in your .env file")
    st.stop()

client = OpenAI(api_key=api_key)

# Header
st.markdown(
    '<div class="title">🤖 AI Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Powered by OpenAI</div>',
    unsafe_allow_html=True
)

st.divider()

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")

    model = st.selectbox(
        "Select Model",
        ["gpt-5.5", "gpt-5.4", "gpt-5.4-mini"]
    )

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.info("Ask me anything!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Ask your question here..."):

    # Display user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate AI response
    with st.chat_message("assistant"):

        try:
            with st.spinner("AI is thinking..."):

                response = client.responses.create(
                    model=model,
                    instructions=(
                        "You are a helpful AI assistant. "
                        "Give clear and easy-to-understand answers."
                    ),
                    input=st.session_state.messages
                )

                answer = response.output_text

            st.markdown(answer)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })

        except Exception as e:
            st.error(f"Error: {str(e)}")
