from fastapi import FastAPI

app = FastAPI()

agent_card = {
    "name": "Hotel Agent",
    "description": "Finds hotels.",
    "version": "1.0",
    "capabilities": [
        "hotel"
    ],
    "endpoint": "http://localhost:8002/invoke"
}

@app.get("/agent-card")
def card():
    return agent_card

@app.post("/invoke")
def invoke(request: dict):
    city = request["city"]

    return {
        "hotel": f"Taj Hotel in {city}"
    }