# HR Agent Evaluation with LangChain + DeepEval

Put your single HR PDF in `data/hr_policy.pdf`.

Install:
```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your OpenAI API key.

Run:
```bash
python ingest.py
python main.py
```

The project evaluates:
- Faithfulness
- Relevance
- Correctness
- Groundedness

Groundedness is reported using the Faithfulness metric because they represent the same RAG concept in this setup.
