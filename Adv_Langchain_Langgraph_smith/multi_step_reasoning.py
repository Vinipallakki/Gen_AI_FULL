
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4.1",
    temperature=0
)

parser = StrOutputParser()


# STEP 1: Classify the incident
classification_prompt = ChatPromptTemplate.from_template("""
You are a production reliability engineer.

Analyze this incident and identify:
- Severity
- Affected service
- Symptoms
- Immediate risks

Incident details:
{incident}
""")

classification_chain = classification_prompt | llm | parser


# STEP 2: Analyze possible root causes
root_cause_prompt = ChatPromptTemplate.from_template("""
You are a senior site reliability engineer.

Analyze the incident using the classification below.

Incident:
{incident}

Classification:
{classification}

Provide:
- Three plausible root causes
- Evidence supporting each cause
- Metrics or logs needed to confirm each cause

Do not claim a root cause is confirmed without evidence.
""")

root_cause_chain = root_cause_prompt | llm | parser


# STEP 3: Recommend remediation
remediation_prompt = ChatPromptTemplate.from_template("""
You are a production incident response engineer.

Incident:
{incident}

Root-cause analysis:
{root_causes}

Recommend:
1. Immediate diagnostic checks
2. Safe mitigation steps
3. Long-term preventive actions

Do not recommend destructive actions without validation.
""")

remediation_chain = remediation_prompt | llm | parser


# STEP 4: Generate the final report
report_prompt = ChatPromptTemplate.from_template("""
Create a concise engineering incident report.

Incident:
{incident}

Classification:
{classification}

Root-cause analysis:
{root_causes}

Remediation plan:
{remediation}

Include:
- Incident summary
- Severity and impact
- Suspected causes
- Immediate actions
- Preventive measures

Clearly distinguish confirmed facts from hypotheses.
""")

report_chain = report_prompt | llm | parser


# Execute the complete multi-step workflow
def analyze_incident(incident):

    print("\nSTEP 1: CLASSIFICATION")
    classification = classification_chain.invoke({
        "incident": incident
    })
    print(classification)

    print("\nSTEP 2: ROOT-CAUSE ANALYSIS")
    root_causes = root_cause_chain.invoke({
        "incident": incident,
        "classification": classification
    })
    print(root_causes)

    print("\nSTEP 3: REMEDIATION PLAN")
    remediation = remediation_chain.invoke({
        "incident": incident,
        "root_causes": root_causes
    })
    print(remediation)

    print("\nSTEP 4: FINAL REPORT")
    report = report_chain.invoke({
        "incident": incident,
        "classification": classification,
        "root_causes": root_causes,
        "remediation": remediation
    })
    print(report)

    return report


incident = """
Production API latency increased from 200 ms to 3 seconds.
The HTTP 5xx error rate reached 15%.
CPU utilization is 92%.
Database query latency has also increased.
The issue started 10 minutes after a new application deployment.
"""

final_report = analyze_incident(incident)
