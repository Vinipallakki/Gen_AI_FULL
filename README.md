# LangChain PDF Bot

## Run locally

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

## Docker

```bash
docker build -t langchain-pdf-bot .
docker run -p 8000:8000 langchain-pdf-bot
```

## API

Home:
http://127.0.0.1:8000

Ask:
http://127.0.0.1:8000/ask?query=what is this pdf about
