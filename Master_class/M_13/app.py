
# import streamlit as st
# from openai import OpenAI
# from dotenv import load_dotenv
# import os

# load_dotenv()

# api_key = os.getenv("OPENAI_API_KEY")

# st.set_page_config(
#     page_title="AI IT Support Assistant",
#     page_icon="🤖"
# )

# st.title("🤖 AI IT Support Assistant")
# st.write("Describe your IT issue and AI will help troubleshoot it.")

# if not api_key:
#     st.error("Please configure your OpenAI API key in .env")
#     st.stop()

# client = OpenAI(api_key=api_key)

# issue = st.text_area(
#     "Describe your IT problem",
#     placeholder="Example: My laptop is connected to Wi-Fi but internet is not working..."
# )

# if st.button("Analyze Issue"):

#     if not issue.strip():
#         st.warning("Please enter an IT problem.")
#     else:
#         prompt = f"""
#         You are an experienced IT support engineer.

#         Analyze this IT issue:

#         {issue}

#         Provide the response in this format:

#         1. Issue Category
#         2. Priority (Low, Medium, High)
#         3. Possible Root Causes
#         4. Troubleshooting Steps
#         5. Recommended Solution

#         Give simple, practical steps.
#         Do not claim that you have actually fixed the issue.
#         """

#         try:
#             with st.spinner("AI is analyzing your issue..."):

#                 response = client.responses.create(
#                     model="gpt-4.1-mini",
#                     input=prompt
#                 )

#             st.success("Analysis completed!")

#             st.markdown(response.output_text)

#         except Exception as e:
#             st.error(f"Something went wrong: {e}")



import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

st.set_page_config(
    page_title="MicroDegree | IT Support",
    page_icon="🤖",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #eef2ff 0%, #f8faff 50%, #f3e8ff 100%);
}

.block-container {
    padding-top: 2rem;
    max-width: 1200px;
}

.hero {
    background: linear-gradient(120deg, #4338ca, #7c3aed, #2563eb);
    padding: 35px;
    border-radius: 22px;
    color: white;
    margin-bottom: 25px;
    box-shadow: 0 10px 30px rgba(79,70,229,0.2);
}

.hero h1 {
    color: white;
    font-size: 36px;
    margin-bottom: 8px;
}

.hero p {
    color: #e0e7ff;
    font-size: 16px;
}

.section-title {
    color: #312e81;
    font-size: 23px;
    font-weight: 750;
    margin-top: 20px;
    margin-bottom: 12px;
}

.info-card {
    background: white;
    padding: 22px;
    border-radius: 16px;
    border: 1px solid #e0e7ff;
    box-shadow: 0 5px 18px rgba(79,70,229,0.07);
    min-height: 125px;
}

.info-card h3 {
    color: #4f46e5;
    font-size: 17px;
}

.info-card p {
    color: #475569;
    font-size: 14px;
}

.stTextArea textarea {
    background: white !important;
    border: 1px solid #c7d2fe !important;
    border-radius: 12px !important;
    color: #1e293b !important;
}

.stButton > button {
    background: linear-gradient(90deg, #4f46e5, #7c3aed);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 12px 25px;
    font-weight: 700;
    min-height: 48px;
    transition: 0.2s;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #3730a3, #6d28d9);
    color: white;
    border: none;
    transform: translateY(-2px);
}

.result-card {
    background: white;
    padding: 24px;
    border-radius: 16px;
    border-left: 5px solid #6366f1;
    box-shadow: 0 5px 20px rgba(79,70,229,0.08);
    margin: 12px 0;
    color: #1e293b;
}

div[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #312e81, #4f46e5);
}

div[data-testid="stSidebar"] * {
    color: white;
}

hr {
    border-color: #e0e7ff;
}
</style>
""", unsafe_allow_html=True)


# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.markdown("# 🤖 MicroDegree AI")
    st.caption("INTELLIGENT IT SUPPORT")

    st.divider()

    st.markdown("### 🚀 Quick Examples")

    examples = {
        "🌐 Network Issue":
            "My laptop is connected to Wi-Fi but I cannot access the internet.",

        "🖥️ Server Issue":
            "Our Linux server CPU utilization is continuously above 95%.",

        "🗄️ Database Issue":
            "My application is unable to connect to PostgreSQL. It shows connection timeout.",

        "🔐 Login Issue":
            "I am unable to log in to my company application even with the correct password."
    }

    if "issue" not in st.session_state:
        st.session_state.issue = ""

    for name, value in examples.items():
        if st.button(name, use_container_width=True):
            st.session_state.issue = value

    st.divider()
    st.markdown("### ✨ Features")
    st.markdown("""
    - AI-powered troubleshooting
    - Issue classification
    - Priority detection
    - Root cause analysis
    - Suggested solutions
    """)

    st.divider()
    st.caption("Powered by Python + OpenAI")


# ---------------- MAIN HEADER ----------------

st.markdown("""
<div class="hero">
    <h1>🤖 Nexa AI Support Assistant</h1>
    <p>Your intelligent IT troubleshooting companion.
    Describe your issue and let AI find possible solutions.</p>
</div>
""", unsafe_allow_html=True)


# ---------------- FEATURE CARDS ----------------

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="info-card">
        <h3>⚡ Smart Analysis</h3>
        <p>Understand IT issues using AI.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-card">
        <h3>🎯 Root Cause</h3>
        <p>Identify possible reasons behind errors.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="info-card">
        <h3>🛠️ Solutions</h3>
        <p>Get practical troubleshooting steps.</p>
    </div>
    """, unsafe_allow_html=True)


# ---------------- INPUT SECTION ----------------

st.markdown(
    '<div class="section-title">💬 Describe Your IT Problem</div>',
    unsafe_allow_html=True
)

issue = st.text_area(
    "Enter your issue",
    key="issue",
    height=130,
    placeholder="Example: My laptop is connected to Wi-Fi, but internet is not working..."
)

analyze = st.button(
    "🚀 Analyze My Issue",
    use_container_width=True
)


# ---------------- AI ANALYSIS ----------------

if analyze:

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        st.error("Please configure OPENAI_API_KEY in your .env file.")

    elif not issue.strip():
        st.warning("Please describe your IT problem first.")

    else:
        client = OpenAI(api_key=api_key)

        prompt = f"""
        You are a senior IT support engineer.

        Analyze this issue:
        {issue}

        Return a structured response with:

        1. Issue Category
        2. Priority: Low, Medium or High
        3. Possible Root Causes
        4. Troubleshooting Steps
        5. Recommended Solution

        Give clear, practical instructions.
        Do not claim that you have actually fixed the issue.
        Mention any safety precautions for risky operations.
        """

        try:
            with st.spinner("🧠 Nexa AI is analyzing your issue..."):

                response = client.responses.create(
                    model="gpt-4.1-mini",
                    input=prompt
                )

                result = response.output_text

            st.markdown("---")
            st.markdown(
                '<div class="section-title">📊 AI Analysis Report</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f'<div class="result-card">{result.replace(chr(10), "<br>")}</div>',
                unsafe_allow_html=True
            )

            st.success("Analysis completed successfully!")

        except Exception as e:
            st.error(f"Error while analyzing: {e}")


# ---------------- FOOTER ----------------

st.markdown("---")
st.markdown(
    "<center style='color:#64748b;'>"
    "Built with ❤️ using Python, Streamlit & Generative AI"
    "</center>",
    unsafe_allow_html=True
)
