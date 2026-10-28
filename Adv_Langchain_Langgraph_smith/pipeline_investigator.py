
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Initialize the LLM
llm = ChatOpenAI(
    model="gpt-4.1",
    temperature=0
)

parser = StrOutputParser()


# --------------------------------------------------
# STEP 1: Extract evidence from the incident
# --------------------------------------------------

evidence_prompt = ChatPromptTemplate.from_template("""
You are a senior data engineer investigating a failed ETL pipeline.

Analyze the incident information below.

Incident:
{incident}

Identify:
1. Confirmed facts
2. Symptoms
3. Recent changes
4. Missing information needed for diagnosis

Do not invent facts or declare an unverified root cause.
""")

evidence_chain = evidence_prompt | llm | parser


# --------------------------------------------------
# STEP 2: Diagnose the likely root cause
# --------------------------------------------------

diagnosis_prompt = ChatPromptTemplate.from_template("""
You are a data platform reliability engineer.

Investigate the incident using the supplied evidence.

Original incident:
{incident}

Evidence analysis:
{evidence}

Identify:
1. Most likely root cause
2. Evidence supporting it
3. Alternative explanations
4. Checks required to confirm the diagnosis

Distinguish likely causes from confirmed causes.
""")

diagnosis_chain = diagnosis_prompt | llm | parser


# --------------------------------------------------
# STEP 3: Create a safe remediation plan
# --------------------------------------------------

remediation_prompt = ChatPromptTemplate.from_template("""
You are responsible for safely recovering a production ETL pipeline.

Incident:
{incident}

Evidence:
{evidence}

Diagnosis:
{diagnosis}

Recommend:
1. Immediate diagnostic checks
2. Code or configuration changes
3. How to handle invalid records
4. How to recover missing records safely
5. Tests and monitoring to prevent recurrence

Do not silently discard records.
Do not recommend rerunning the entire pipeline blindly.
Consider duplicate prevention and data integrity.
""")

remediation_chain = remediation_prompt | llm | parser


# --------------------------------------------------
# STEP 4: Generate the final incident report
# --------------------------------------------------

report_prompt = ChatPromptTemplate.from_template("""
Prepare a concise production incident report for engineers.

ORIGINAL INCIDENT:
{incident}

EVIDENCE:
{evidence}

ROOT-CAUSE ANALYSIS:
{diagnosis}

REMEDIATION PLAN:
{remediation}

Include:
- Incident summary
- Impact
- Evidence and likely root cause
- Immediate actions
- Safe recovery plan
- Validation and preventive measures

Clearly label unconfirmed hypotheses.
Do not claim that any fix was executed or verified.
""")

report_chain = report_prompt | llm | parser


# --------------------------------------------------
# MAIN MULTI-STEP REASONING WORKFLOW
# --------------------------------------------------

def investigate_pipeline(incident):

    print("\n===== STEP 1: EVIDENCE =====")

    evidence = evidence_chain.invoke({
        "incident": incident
    })
    print(evidence)

    print("\n===== STEP 2: DIAGNOSIS =====")

    diagnosis = diagnosis_chain.invoke({
        "incident": incident,
        "evidence": evidence
    })
    print(diagnosis)

    print("\n===== STEP 3: REMEDIATION =====")

    remediation = remediation_chain.invoke({
        "incident": incident,
        "evidence": evidence,
        "diagnosis": diagnosis
    })
    print(remediation)

    print("\n===== STEP 4: FINAL REPORT =====")

    report = report_chain.invoke({
        "incident": incident,
        "evidence": evidence,
        "diagnosis": diagnosis,
        "remediation": remediation
    })
    print(report)

    return {
        "evidence": evidence,
        "diagnosis": diagnosis,
        "remediation": remediation,
        "report": report
    }


# --------------------------------------------------
# SAMPLE PRODUCTION INCIDENT
# --------------------------------------------------

incident = """
Pipeline: customer_transactions_etl

Expected records: 1,000,000
Records loaded: 620,000
Records not loaded: 380,000

Source database status: Healthy
BigQuery status: Available

Error log:
Invalid timestamp format

Recent change:
The source team changed the timestamp format.

Business impact:
The daily transaction reporting table may be incomplete.
"""

result = investigate_pipeline(incident)
