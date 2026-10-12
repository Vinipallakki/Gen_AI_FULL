from fastapi import FastAPI

app = FastAPI()

agent_card = {
    "name": "Weather Agent",
    "description": "Provides weather information.",
    "version": "1.0",
    "capabilities": [
        "weather"
    ],
    "endpoint": "http://localhost:8001/invoke"
}

@app.get("/agent-card")
def card():
    return agent_card

@app.post("/invoke")
def invoke(request: dict):
    city = request["city"]

    return {
        "weather": f"28°C and Cloudy in {city}"
    }