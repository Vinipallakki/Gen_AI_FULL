# AI IT Support --- Production-Ready Multi-Agent AI System

A small production-oriented **Multi-Agent AI IT Support system** built
with **OpenAI, FastAPI, Streamlit, Chroma, Redis, Celery, Docker
Compose, Prometheus, and GCP Compute Engine**.

The project demonstrates how to take an AI application from local
development to a containerized cloud deployment while implementing
caching, rate limiting, background execution, retries, timeouts, health
checks, logging, request tracing, and monitoring.

------------------------------------------------------------------------

## 1. Project Goals

This project is designed around the following production AI engineering
requirements:

-   Containerize applications using Docker
-   Deploy AI systems on cloud environments
-   Implement background execution
-   Implement caching strategies
-   Implement rate limiting
-   Build async APIs and workers
-   Monitor production AI systems
-   Apply production reliability practices
-   Deploy a multi-agent AI application
-   Use RAG with Chroma
-   Use OpenAI for routing and answer generation

------------------------------------------------------------------------

# 2. High-Level Architecture

``` text
                         USER
                           |
                           v
                    +-------------+
                    |  Streamlit  |
                    |    :8501    |
                    +------+------+
                           |
                           v
                    +-------------+
                    |   FastAPI   |
                    |    :8000    |
                    +------+------+
                           |
             +-------------+-------------+
             |             |             |
             v             v             v
        Rate Limit     Request ID    Redis Cache
             |                           |
             +-------------+-------------+
                           |
                           v
                    +-------------+
                    | Router Agent|
                    +------+------+
                           |
              +------------+-------------+
              |            |              |
              v            v              v
       Knowledge Agent  Troubleshooting  Ticket Agent
              |            Agent             |
              v                              v
           Chroma                         Celery
            / RAG                           |
              |                             v
              v                           Redis
           OpenAI                           |
                                            v
                                      Background Worker

                           |
                           v
                    +-------------+
                    |  Prometheus |
                    |   /metrics  |
                    +-------------+
```

------------------------------------------------------------------------

# 3. Main Components

  Component            Purpose
  -------------------- -----------------------------------------------------
  OpenAI               Routing, reasoning, response generation, embeddings
  FastAPI              Backend API
  Streamlit            User interface
  Chroma               Vector database for RAG
  Redis                Cache + Celery broker/result backend
  Celery               Background task execution
  Docker               Containerization
  Docker Compose       Multi-container orchestration
  Prometheus           Application metrics
  GCP Compute Engine   Cloud deployment

------------------------------------------------------------------------

# 4. Agents

The system contains a router and specialized agents.

                    User Question
                         |
                         v
                   Router Agent
                         |
             +-----------+-----------+
             |           |           |
             v           v           v
         Knowledge  Troubleshooting Ticket
           Agent        Agent       Agent
```

## 4.1 Router Agent

The router determines the category of the question.

Example:

``` text
"What is the password policy?"
        |
        v
    knowledge
```

``` text
"My VPN is not connecting."
        |
        v
    troubleshooting
```

``` text
"Create a support ticket."
        |
        v
       ticket
```

The router uses OpenAI.

------------------------------------------------------------------------

# 5. Knowledge Agent --- RAG Flow

Knowledge questions use Chroma.

Example:

``` text
User:
"What is the company password policy?"
```

Flow:

``` text
User
 |
 v
FastAPI
 |
 v
Router Agent
 |
 v
Knowledge Agent
 |
 v
Chroma
 |
 v
Relevant document chunks
 |
 v
OpenAI
 |
 v
Answer
```

The knowledge base contains IT support documents such as:

``` text
knowledge/
├── vpn.txt
├── wifi.txt
├── password.txt
└── laptop.txt
```

------------------------------------------------------------------------

# 6. Chroma Ingestion Flow

Knowledge documents are indexed using the ingestion script.

Run locally:

``` powershell
python -m scripts.ingest
```

Inside Docker:

``` bash
docker compose exec api python -m scripts.ingest
```

Flow:

``` text
Knowledge Documents
       |
       v
Document Loader
       |
       v
Text Chunks
       |
       v
OpenAI Embeddings
       |
       v
Chroma Vector Store
```

The Chroma database is persisted through Docker storage so that
recreating containers does not automatically remove the vector data.

------------------------------------------------------------------------

# 7. Troubleshooting Agent

Example:

``` text
"My VPN is not connecting. What should I do?"
```

Flow:

``` text
User
 |
 v
FastAPI
 |
 v
Router
 |
 v
Troubleshooting Agent
 |
 v
OpenAI
 |
 v
Response
```

This is a read-only AI request and can be cached.

------------------------------------------------------------------------

# 8. Ticket Agent

Ticket creation is a side-effect and therefore is intentionally **not
cached**.

Example:

``` text
"Create an IT support ticket because my laptop screen is damaged."
```

Flow:

``` text
User
 |
 v
FastAPI
 |
 v
Router
 |
 v
Ticket Agent
 |
 v
Celery Task
 |
 v
Redis
 |
 v
Celery Worker
 |
 v
Ticket Created
```

The API returns a task ID immediately.

Example:

``` text
agent    : ticket_agent
message  : IT support ticket creation started.
task_id  : 7b15f19d-b941-4103-8056-14511bfbbec6
category : ticket
cached   : False
```

The worker processes the task asynchronously.

------------------------------------------------------------------------

# 9. Why Ticket Requests Are Not Cached

Caching a side-effect would be dangerous.

Incorrect design:

``` text
User 1
 |
 v
Create ticket
 |
 v
IT-1001
 |
 v
Cache

User 2
 |
 v
Create ticket
 |
 v
Cache
 |
 v
IT-1001   <-- WRONG
```

Correct design:

``` text
Read-only request
        |
        v
      Cache
```

But:

``` text
Ticket creation
        |
        v
    Execute task
```

This is an important production reliability principle.

------------------------------------------------------------------------

# 10. Redis Caching

Redis is used as the application cache.

Flow for a cached request:

``` text
User
 |
 v
FastAPI
 |
 v
Redis
 |
 +---- Cache HIT ----> Return response
 |
 +---- Cache MISS
          |
          v
       AI Agent
          |
          v
        OpenAI
          |
          v
       Response
          |
          v
        Redis
```

Example:

First request:

``` text
cached : False
```

Second identical request:

``` text
cached : True
```

This reduces:

-   OpenAI API calls
-   latency
-   cost
-   unnecessary computation

------------------------------------------------------------------------

# 11. Cache Design Principle

Only appropriate read-only responses should be cached.

Examples:

``` text
"What is the password policy?"
        |
        v
       Cache
```

``` text
"My VPN is not connecting."
        |
        v
       Cache
```

But:

``` text
"Create a support ticket."
        |
        v
       NO CACHE
```

------------------------------------------------------------------------

# 12. Rate Limiting

The FastAPI support endpoint uses rate limiting.

Example configuration:

``` text
10 requests/minute
```

Purpose:

``` text
User
 |
 +---- Request 1
 +---- Request 2
 +---- ...
 +---- Request 10
 |
 +---- Request 11
          |
          v
      Rate Limited
```

Rate limiting protects the API from:

-   accidental request floods
-   abusive clients
-   excessive OpenAI usage
-   unnecessary cost

------------------------------------------------------------------------

# 13. Async Architecture

FastAPI uses asynchronous endpoints.

Example conceptual flow:

``` text
Request
   |
   v
async FastAPI
   |
   +---- OpenAI async request
   |
   +---- Redis async request
   |
   +---- Other async operations
```

This allows the API server to handle other requests while waiting on
network operations.

------------------------------------------------------------------------

# 14. Background Execution with Celery

Celery handles long-running tasks.

Example:

``` text
FastAPI
 |
 | create task
 v
Celery
 |
 v
Redis
 |
 v
Worker
 |
 | process asynchronously
 v
Ticket created
```

The API does not need to block while the worker performs the operation.

Start a worker locally:

``` powershell
celery -A app.workers.ticket_worker.celery_app worker --loglevel=info --pool=solo
```

For Docker:

``` bash
docker compose logs -f worker
```

------------------------------------------------------------------------

# 15. Celery Task Example

A ticket task can simulate a longer operation:

``` text
Starting ticket creation
        |
        v
       5 sec
        |
        v
Ticket created: IT-XXXXXXXXXX
```

Worker logs look like:

``` text
Task app.workers.ticket_worker.create_ticket[...] received
Starting ticket creation
Ticket created: IT-XXXXXXXXXX
Task ... succeeded
```

------------------------------------------------------------------------

# 16. OpenAI Integration

The application uses the OpenAI Python SDK asynchronously.

The OpenAI service:

1.  Sends a system prompt
2.  Sends the user prompt
3.  Receives the model response
4.  Returns the text to the agent

The project uses:

``` text
OPENAI_MODEL=gpt-5.5
```

and:

``` text
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

------------------------------------------------------------------------

# 17. OpenAI Reliability

The OpenAI client is configured with a timeout.

Example:

``` text
timeout = 30 seconds
```

The application also implements application-level retry behavior.

Conceptually:

``` text
Attempt 1
   |
   +---- failure
          |
       wait 1 sec
          |
Attempt 2
   |
   +---- failure
          |
       wait 2 sec
          |
Attempt 3
   |
   +---- success
```

This helps handle temporary failures.

Retries should not be blindly applied to every operation. Side-effecting
operations need additional care such as idempotency.

------------------------------------------------------------------------

# 18. Request IDs

Every HTTP request receives a unique request ID.

Example:

``` text
request_id=cc1c58bc-7d94-41ad-9da7-3769b8bdd57c
```

The logs can then be traced using the same ID.

Example:

``` text
request_started
request_id=cc1c58bc-7d94-41ad-9da7-3769b8bdd57c

Router selected: troubleshooting

OpenAI request attempt=1

request_completed
request_id=cc1c58bc-7d94-41ad-9da7-3769b8bdd57c
status=200
duration=11.013s
```

This is useful when debugging production requests.

------------------------------------------------------------------------

# 19. Structured Logging

The API produces logs containing information such as:

``` text
timestamp
log level
module
request ID
agent
status
duration
```

Example:

``` text
2026-08-17 06:59:17 | INFO | app.main |
request_started request_id=... method=POST path=/support
```

This makes production troubleshooting easier.

------------------------------------------------------------------------

# 20. Health Checks

The application exposes:

``` text
GET /health
```

The health endpoint checks the application and dependent services such
as Redis.

Expected response:

``` json
{
  "status": "healthy",
  "services": {
    "api": "healthy",
    "redis": "healthy",
    "chroma": "healthy"
  }
}
```

A degraded dependency should not be hidden behind a generic "healthy"
status.

------------------------------------------------------------------------

# 21. Prometheus Monitoring

Prometheus metrics are exposed at:

``` text
/metrics
```

Local:

``` text
http://localhost:8000/metrics
```

GCP:

``` text
http://YOUR_VM_EXTERNAL_IP:8000/metrics
```

Metrics include items such as:

``` text
http_requests_total
http_request_duration_seconds
process_cpu_seconds_total
process_resident_memory_bytes
process_open_fds
```

This project intentionally does not require Grafana.

Prometheus metrics are sufficient to demonstrate monitoring fundamentals
for this project.

------------------------------------------------------------------------

# 22. Docker Architecture

The project is containerized using Docker Compose.

``` text
Docker Compose
 |
 +---- api
 |
 +---- worker
 |
 +---- frontend
 |
 +---- redis
```

Conceptually:

``` text
+----------------------- Docker Compose ----------------------+
|                                                            |
|  +-----------+       +-----------+                         |
|  | Streamlit | ----> |  FastAPI  |                         |
|  |   :8501   |       |   :8000   |                         |
|  +-----------+       +-----+-----+                         |
|                            |                               |
|                            v                               |
|                        +-------+                           |
|                        | Redis |                           |
|                        +---+---+                           |
|                            |                               |
|                            v                               |
|                       +---------+                          |
|                       | Celery  |                          |
|                       | Worker  |                          |
|                       +---------+                          |
|                                                            |
+------------------------------------------------------------+
```

------------------------------------------------------------------------

# 23. Local Development

## Prerequisites

Install:

-   Python 3.11+
-   Docker Desktop
-   Git
-   OpenAI API key

------------------------------------------------------------------------

# 24. Local Environment File

Create `.env`:

``` env
OPENAI_API_KEY=YOUR_OPENAI_API_KEY
OPENAI_MODEL=gpt-5.5
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
REDIS_URL=redis://localhost:6379/0
CACHE_TTL=300
```

Never commit `.env`.

Add:

``` text
.env
__pycache__/
*.pyc
.venv/
venv/
chroma_db/
```

to `.gitignore`.

------------------------------------------------------------------------

# 25. Run Redis Locally

If running FastAPI directly on Windows:

``` powershell
docker run -d --name ai-support-redis -p 6379:6379 redis:7-alpine
```

Check:

``` powershell
docker ps
```

Test Redis:

``` powershell
docker exec -it ai-support-redis redis-cli ping
```

Expected:

``` text
PONG
```

------------------------------------------------------------------------

# 26. Fixing the Windows Redis Hostname

When FastAPI runs directly on Windows:

``` env
REDIS_URL=redis://localhost:6379/0
```

When FastAPI runs inside Docker Compose:

``` env
REDIS_URL=redis://redis:6379/0
```

Why?

Inside Docker Compose, `redis` is the service hostname.

``` text
API container
      |
      +---- redis:6379
                 |
                 v
              Redis
```

Windows does not automatically resolve the Docker Compose service name
`redis`.

------------------------------------------------------------------------

# 27. Local Chroma Ingestion

Run:

``` powershell
python -m scripts.ingest
```

Expected documents:

``` text
Indexed: vpn.txt
Indexed: wifi.txt
Indexed: password.txt
Indexed: laptop.txt
```

------------------------------------------------------------------------

# 28. Run FastAPI Locally

``` powershell
uvicorn app.main:app --reload --port 8000
```

Swagger:

``` text
http://127.0.0.1:8000/docs
```

Health:

``` text
http://127.0.0.1:8000/health
```

Metrics:

``` text
http://127.0.0.1:8000/metrics
```

------------------------------------------------------------------------

# 29. Test FastAPI on Windows PowerShell

Do not use Linux-style `curl` syntax directly in PowerShell because
`curl` may map to `Invoke-WebRequest`.

Recommended:

``` powershell
$body = @{
    question = "My VPN is not connecting. What should I do?"
} | ConvertTo-Json

$response = Invoke-RestMethod `
    -Uri "http://127.0.0.1:8000/support" `
    -Method POST `
    -ContentType "application/json" `
    -Body $body

$response | Format-List
```

Alternatively use the actual executable:

``` powershell
curl.exe -X POST "http://127.0.0.1:8000/support" `
  -H "Content-Type: application/json" `
  --data-raw '{"question":"My VPN is not connecting. What should I do?"}'
```

------------------------------------------------------------------------

# 30. Run Celery Locally

``` powershell
celery -A app.workers.ticket_worker.celery_app worker --loglevel=info --pool=solo
```

Then create a ticket from the API.

The worker should show:

``` text
Task received
Starting ticket creation
Ticket created
Task succeeded
```

------------------------------------------------------------------------

# 31. Run the Complete Application with Docker

Once Docker Compose is configured:

``` powershell
docker compose up --build
```

Check:

``` powershell
docker ps
```

Expected services:

``` text
ai-support-api
ai-support-worker
ai-support-redis
ai-support-frontend
```

------------------------------------------------------------------------

# 32. Dockerized Local URLs

FastAPI:

``` text
http://localhost:8000
```

Swagger:

``` text
http://localhost:8000/docs
```

Health:

``` text
http://localhost:8000/health
```

Metrics:

``` text
http://localhost:8000/metrics
```

Streamlit:

``` text
http://localhost:8501
```

------------------------------------------------------------------------

# 33. Docker Logs

API:

``` bash
docker compose logs -f api
```

Worker:

``` bash
docker compose logs -f worker
```

Frontend:

``` bash
docker compose logs -f frontend
```

Redis:

``` bash
docker compose logs -f redis
```

------------------------------------------------------------------------

# 34. Dockerized Chroma Ingestion

Run:

``` bash
docker compose exec api python -m scripts.ingest
```

Expected:

``` text
Indexed: vpn.txt
Indexed: wifi.txt
Indexed: password.txt
Indexed: laptop.txt
```

------------------------------------------------------------------------

# 35. GCP Deployment

The cloud deployment uses:

``` text
GCP Compute Engine
        |
        v
Ubuntu VM
        |
        v
Docker
        |
        v
Docker Compose
        |
        +---- FastAPI
        +---- Streamlit
        +---- Celery
        +---- Redis
```

For this learning project, a small Compute Engine VM such as an
`e2-medium` can be used as a starting point.

------------------------------------------------------------------------

# 36. Create the GCP VM

In Google Cloud Console:

``` text
Compute Engine
    |
    v
VM Instances
    |
    v
Create Instance
```

Example:

``` text
Name: ai-it-support-vm
OS: Ubuntu 24.04 LTS
Machine: e2-medium
Disk: 20–30 GB
```

Choose a region close to your users.

------------------------------------------------------------------------

# 37. SSH Into the VM

Use the GCP Console SSH button.

Then verify:

``` bash
uname -a
```

------------------------------------------------------------------------

# 38. Install Docker on GCP VM

``` bash
sudo apt update
sudo apt install -y docker.io docker-compose-v2 git
```

Verify:

``` bash
docker --version
docker compose version
```

Allow the current user to run Docker:

``` bash
sudo usermod -aG docker $USER
```

Then:

``` bash
newgrp docker
```

Test:

``` bash
docker ps
```

------------------------------------------------------------------------

# 39. Clone the Repository

On the VM:

``` bash
git clone YOUR_GITHUB_REPOSITORY
```

Then:

``` bash
cd ai-it-support
```

Verify:

``` bash
ls
```

Expected project structure:

``` text
Dockerfile
docker-compose.yml
requirements.txt
app/
frontend/
knowledge/
scripts/
```

------------------------------------------------------------------------

# 40. Create GCP `.env`

Do NOT commit the production `.env`.

If `vim` or `vi` is not installed, install nano:

``` bash
sudo apt update
sudo apt install -y nano
```

Then:

``` bash
nano .env
```

For Docker Compose on GCP:

``` env
OPENAI_API_KEY=YOUR_REAL_OPENAI_API_KEY
OPENAI_MODEL=gpt-5.5
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
REDIS_URL=redis://redis:6379/0
CACHE_TTL=300
```

Save:

``` text
CTRL + O
ENTER
CTRL + X
```

Verify non-secret values:

``` bash
grep -E 'OPENAI_MODEL|OPENAI_EMBEDDING_MODEL|REDIS_URL|CACHE_TTL' .env
```

Never print or share the OpenAI API key.

------------------------------------------------------------------------

# 41. GCP Redis Security

Redis should not be publicly exposed.

Do NOT expose:

``` text
6379
```

to the internet.

The desired network is:

``` text
Internet
   |
   +---- 8000 ---> FastAPI
   |
   +---- 8501 ---> Streamlit
                  |
                  v
             Internal Docker Network
                  |
                  v
                Redis
                 6379
```

The API and Celery worker can reach Redis through:

``` text
redis:6379
```

without publishing port 6379.

------------------------------------------------------------------------

# 42. Build on GCP

From the project directory:

``` bash
docker compose build
```

------------------------------------------------------------------------

# 43. Start on GCP

``` bash
docker compose up -d
```

Check:

``` bash
docker ps
```

Expected:

``` text
ai-support-api
ai-support-worker
ai-support-redis
ai-support-frontend
```

------------------------------------------------------------------------

# 44. Ingest Chroma on GCP

Run:

``` bash
docker compose exec api python -m scripts.ingest
```

This creates/updates the vector database used by the Knowledge Agent.

------------------------------------------------------------------------

# 45. GCP Firewall

For the learning deployment, expose:

``` text
TCP 8000
TCP 8501
```

Do not expose:

``` text
TCP 6379
```

The firewall rule can target the VM with a network tag such as:

``` text
ai-support
```

------------------------------------------------------------------------

# 46. Access the Deployed Application

Find the VM external IP in:

``` text
Compute Engine
    |
    v
VM Instances
```

Then:

Streamlit:

``` text
http://YOUR_VM_EXTERNAL_IP:8501
```

FastAPI:

``` text
http://YOUR_VM_EXTERNAL_IP:8000
```

Swagger:

``` text
http://YOUR_VM_EXTERNAL_IP:8000/docs
```

Health:

``` text
http://YOUR_VM_EXTERNAL_IP:8000/health
```

Metrics:

``` text
http://YOUR_VM_EXTERNAL_IP:8000/metrics
```

------------------------------------------------------------------------

# 47. Production Request Flow

Example:

``` text
User
 |
 v
Streamlit
 |
 v
FastAPI
 |
 +--> Generate Request ID
 |
 +--> Rate Limit
 |
 +--> Check Redis Cache
 |
 +---- Cache HIT
 |       |
 |       +----> Return cached answer
 |
 +---- Cache MISS
         |
         v
     Router Agent
         |
         +-------- Knowledge
         |             |
         |             v
         |          Chroma
         |             |
         |             v
         |           OpenAI
         |
         +-------- Troubleshooting
         |             |
         |             v
         |           OpenAI
         |
         +-------- Ticket
                       |
                       v
                    Celery
                       |
                       v
                     Redis
                       |
                       v
                 Celery Worker
                       |
                       v
                  Ticket Created
```

------------------------------------------------------------------------

# 48. Monitoring Flow

``` text
Client
 |
 v
FastAPI
 |
 +---- Request ID
 |
 +---- Structured Logs
 |
 +---- Metrics
 |
 +---- Health Check
 |
 v
Prometheus /metrics
```

Example metrics:

``` text
http_requests_total
http_request_duration_seconds
process_cpu_seconds_total
process_resident_memory_bytes
```

------------------------------------------------------------------------

# 49. Reliability Flow

The application uses multiple reliability mechanisms:

``` text
                 Request
                    |
                    v
              Rate Limiting
                    |
                    v
                Timeout
                    |
                    v
              OpenAI Request
                    |
             +------+------+
             |             |
          Success        Failure
             |             |
             v             v
          Response      Retry
                           |
                           v
                       Backoff
                           |
                           v
                     Retry Again
```

For background jobs:

``` text
FastAPI
   |
   v
Celery
   |
   v
Redis
   |
   v
Worker
   |
   +---- success
   |
   +---- failure -> retry strategy
```

------------------------------------------------------------------------

# 50. Deployment Pattern

The project follows this progression:

``` text
Stage 1
Local Python
     |
     v
FastAPI + Redis
```

``` text
Stage 2
Docker
     |
     v
FastAPI + Redis + Celery + Streamlit
```

``` text
Stage 3
Docker Compose
     |
     v
Complete local multi-service system
```

``` text
Stage 4
GCP Compute Engine
     |
     v
Ubuntu VM
     |
     v
Docker Compose
     |
     v
Cloud-hosted AI system
```

This demonstrates a practical deployment pattern without introducing
Kubernetes complexity.

------------------------------------------------------------------------

# 51. Production Improvements After Basic GCP Deployment

The current deployment is suitable as a learning/portfolio
production-style project, but a real production deployment should
additionally consider:

-   HTTPS
-   Domain name
-   Reverse proxy
-   Authentication/authorization
-   Secret Manager instead of a plain VM `.env`
-   Static IP
-   Automated deployments
-   Centralized logging
-   Alerting
-   Backup strategy
-   Redis persistence strategy
-   Chroma persistence and backup
-   Celery task idempotency
-   OpenAI cost controls
-   Autoscaling
-   Load balancing
-   Resource limits
-   Container restart policies
-   Dependency vulnerability scanning

------------------------------------------------------------------------

# 52. Important Security Rules

Never commit:

``` text
.env
OpenAI API keys
private credentials
service account keys
```

Never expose Redis publicly:

``` text
6379
```

Do not hard-code API keys in Python:

``` python
# WRONG
api_key = "sk-..."
```

Use environment variables:

``` python
settings.OPENAI_API_KEY
```

------------------------------------------------------------------------

# 53. Troubleshooting

## Docker daemon not running

Error:

``` text
failed to connect to the Docker API
```

Solution:

Start Docker Desktop on Windows.

Then:

``` powershell
docker info
```

------------------------------------------------------------------------

## Redis hostname error locally

Error:

``` text
getaddrinfo failed
redis:6379
```

If FastAPI is running directly on Windows, use:

``` env
REDIS_URL=redis://localhost:6379/0
```

If FastAPI runs inside Docker Compose, use:

``` env
REDIS_URL=redis://redis:6379/0
```

------------------------------------------------------------------------

## PowerShell curl error

If:

``` powershell
curl
```

produces an `Invoke-WebRequest` parameter error, use:

``` powershell
Invoke-RestMethod
```

or:

``` powershell
curl.exe
```

------------------------------------------------------------------------

## JSON decode error

Use:

``` powershell
$body = @{
    question = "My VPN is not connecting. What should I do?"
} | ConvertTo-Json

Invoke-RestMethod `
    -Uri "http://127.0.0.1:8000/support" `
    -Method POST `
    -ContentType "application/json" `
    -Body $body
```

------------------------------------------------------------------------

## OpenAI SDK error

If an older code version uses:

``` python
client.responses.create(...)
```

while the installed SDK/client does not expose that API, use the
compatible chat completions call used by this project:

``` python
response = await client.chat.completions.create(
    model=settings.OPENAI_MODEL,
    messages=[
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]
)
```

Then:

``` python
return response.choices[0].message.content
```

------------------------------------------------------------------------

## Docker container logs

Use:

``` bash
docker compose logs -f api
```

``` bash
docker compose logs -f worker
```

``` bash
docker compose logs -f frontend
```

``` bash
docker compose logs -f redis
```

------------------------------------------------------------------------

# 54. Useful Docker Commands

Start:

``` bash
docker compose up -d
```

Build and start:

``` bash
docker compose up -d --build
```

Stop:

``` bash
docker compose down
```

View containers:

``` bash
docker compose ps
```

View logs:

``` bash
docker compose logs -f
```

Restart API:

``` bash
docker compose restart api
```

Restart worker:

``` bash
docker compose restart worker
```

Enter API container:

``` bash
docker compose exec api bash
```

Run ingestion:

``` bash
docker compose exec api python -m scripts.ingest
```

------------------------------------------------------------------------

# 55. End-to-End Test Checklist

## Application

-   [ ] Streamlit opens
-   [ ] FastAPI `/health` works
-   [ ] FastAPI `/docs` works
-   [ ] `/metrics` works

## AI

-   [ ] Troubleshooting question works
-   [ ] Knowledge/RAG question works
-   [ ] Router selects correct agent
-   [ ] OpenAI response works

## RAG

-   [ ] Knowledge files exist
-   [ ] Ingestion completes
-   [ ] Chroma is populated
-   [ ] Knowledge Agent retrieves relevant information

## Cache

-   [ ] First request returns `cached=False`
-   [ ] Second identical read-only request returns `cached=True`
-   [ ] Ticket requests are not cached

## Background

-   [ ] Ticket request returns task ID
-   [ ] Celery worker receives task
-   [ ] Worker creates ticket
-   [ ] Worker logs show success

## Reliability

-   [ ] OpenAI timeout configured
-   [ ] OpenAI retry configured
-   [ ] Rate limiting configured
-   [ ] Request IDs generated
-   [ ] Structured logs available
-   [ ] Health check available

## Monitoring

-   [ ] `/metrics` available
-   [ ] Request metrics appear
-   [ ] Latency metrics appear
-   [ ] Process metrics appear

## Docker

-   [ ] API container running
-   [ ] Worker container running
-   [ ] Redis container running
-   [ ] Streamlit container running
-   [ ] Chroma data persisted

## GCP

-   [ ] Compute Engine VM running
-   [ ] Docker installed
-   [ ] Repository cloned
-   [ ] `.env` created on VM
-   [ ] Docker Compose running
-   [ ] Firewall configured
-   [ ] Redis not publicly exposed
-   [ ] Streamlit accessible
-   [ ] FastAPI accessible
-   [ ] RAG works on VM
-   [ ] Celery works on VM

------------------------------------------------------------------------

# 56. Final Architecture

``` text
                         INTERNET
                            |
                            v
                +-----------------------+
                | GCP Compute Engine VM  |
                |                       |
                |   Docker Compose      |
                |                       |
                | +-------------------+ |
                | |    Streamlit      | |
                | |      :8501        | |
                | +---------+---------+ |
                |           |           |
                | +---------v---------+ |
                | |      FastAPI      | |
                | |       :8000       | |
                | +---------+---------+ |
                |           |           |
                |     +-----+-----+     |
                |     |           |     |
                |     v           v     |
                |   Redis      Chroma   |
                |   Cache        RAG    |
                |     |                 |
                |     v                 |
                |  Celery Worker        |
                |                       |
                +-----------+-----------+
                            |
                            v
                       OpenAI API
```

------------------------------------------------------------------------

# 57. Interview Explanation

A concise explanation of the project:

> I built a containerized multi-agent AI IT Support system using
> FastAPI, Streamlit, OpenAI, Chroma, Redis, and Celery. A router agent
> classifies incoming support requests into knowledge, troubleshooting,
> or ticket workflows. Knowledge queries use Chroma-based RAG,
> troubleshooting requests use OpenAI, and ticket requests are executed
> asynchronously using Celery with Redis as the broker. Redis also
> provides caching for read-only responses. The API includes rate
> limiting, asynchronous execution, OpenAI timeouts and retries, request
> IDs, structured logging, health checks, and Prometheus metrics. The
> complete application is packaged with Docker Compose and deployed to a
> GCP Compute Engine VM.

------------------------------------------------------------------------

# 58. Skills Demonstrated

``` text
Python
FastAPI
Streamlit
OpenAI
Multi-Agent AI
RAG
Chroma
Embeddings
Redis
Caching
Celery
Async Programming
Background Workers
Docker
Docker Compose
Rate Limiting
Retries
Timeouts
Request Tracing
Structured Logging
Health Checks
Prometheus
GCP Compute Engine
Cloud Deployment
Production Reliability
```

------------------------------------------------------------------------

# 59. Project Lifecycle

``` text
                  PROJECT LIFECYCLE

       AI Application Development
                   |
                   v
            Multi-Agent System
                   |
                   v
               RAG + Chroma
                   |
                   v
            FastAPI + Streamlit
                   |
                   v
             Redis Caching
                   |
                   v
           Celery Background Jobs
                   |
                   v
             Rate Limiting
                   |
                   v
           Retry + Timeout
                   |
                   v
          Request IDs + Logging
                   |
                   v
             Health Checks
                   |
                   v
          Prometheus Metrics
                   |
                   v
             Docker Compose
                   |
                   v
          GCP Compute Engine
                   |
                   v
        Cloud-Hosted AI System
```

------------------------------------------------------------------------

## 60. Final Result

The completed project demonstrates the complete journey:

``` text
LOCAL DEVELOPMENT
       |
       v
AI APPLICATION
       |
       v
MULTI-AGENT + RAG
       |
       v
FASTAPI + STREAMLIT
       |
       v
REDIS + CELERY
       |
       v
RELIABILITY
       |
       +---- Retry
       +---- Timeout
       +---- Rate Limit
       +---- Request ID
       +---- Logging
       +---- Health Check
       +---- Metrics
       |
       v
DOCKER COMPOSE
       |
       v
GCP COMPUTE ENGINE
       |
       v
PRODUCTION-STYLE AI SYSTEM
```

This project can be used as a portfolio project and as a practical
demonstration of **AI engineering + backend engineering +
containerization + cloud deployment + production reliability**.
