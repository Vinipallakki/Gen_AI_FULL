from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home(): 
    return {
        "message": "Hello Docker from Vinayaka!"
    }

@app.get("/about")
def about():
    return {
        "course": "Docker Basics"
    }