from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from Gen_AI_FULL.fast_Api.db import SessionLocal, engine
import Gen_AI_FULL.fast_Api.models as models

models.Base.metadata.create_all(bind=engine)

app = FastAPI()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.post("/users/")
def create_user(name: str, age: int, db: Session = Depends(get_db)):
    user = models.User(name=name, age=age)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

@app.get("/users/")
def get_users(db: Session = Depends(get_db)):
    return db.query(models.User).all()

@app.put("/users/{user_id}")
def update_user(user_id: int, name: str, age: int, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.id == user_id).first()
    user.name = name
    user.age = age
    db.commit()
    return user

from fastapi import HTTPException

@app.delete("/users/{user_id}")
def delete_user(user_id: int, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.id == user_id).first()

    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    db.delete(user)
    db.commit()

    return {"message": "Deleted successfully"}




