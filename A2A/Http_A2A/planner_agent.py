from fastapi import FastAPI
import requests

app = FastAPI()

agent_card = {
    "name": "Planner Agent",
    "description": "Plans trips using multiple agents.",
    "version": "1.0",
    "capabilities": [
        "planning"
    ],
    "endpoint": "http://localhost:8000/invoke"
}


@app.get("/agent-card")
def card():
    return agent_card


@app.post("/invoke")
def invoke(request: dict):

    city = request["city"]

    weather = requests.post(
        "http://localhost:8001/invoke",
        json={"city": city}
    ).json()

    hotel = requests.post(
        "http://localhost:8002/invoke",
        json={"city": city}
    ).json()

    return {
        "city": city,
        "weather": weather["weather"],
        "hotel": hotel["hotel"]
    }