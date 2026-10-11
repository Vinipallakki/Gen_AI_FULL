from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {
        "message": "Hello Docker!"
    }

@app.get("/about")
def about():
    return {
        "course": "Docker Basics"
    }