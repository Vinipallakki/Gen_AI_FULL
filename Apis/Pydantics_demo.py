# from pydantic import BaseModel

# class User(BaseModel):
#     name: str
#     age: int

# user = User(name="Vinayak", age="25")  # age is string!

# print(user)
# print(type(user.age))

# from pydantic import BaseModel

# class User(BaseModel):
#     name: str
#     age: int

# user = User(name="Vinayak", age="abc")  # invalid

# from pydantic import BaseModel, field_validator
# from typing import Optional, List


# class User(BaseModel):
#     name: str
#     age: int
#     email: Optional[str] = None
#     tags: List[str] = []

#     @field_validator("age")
#     def validate_age(cls, v):
#         if v < 0:
#             raise ValueError("Age must be positive")
#         return v


# # Example usage
# user = User(
#     name="vinayaka",
#     age="25",              # string → converted to int
#     email="vinayak@mail.com",
#     tags=["ai", "python"]
# )

# print(user)
# print(type(user.age))

from fastapi import FastAPI
from pydantic import BaseModel, EmailStr
from typing import List, Optional

app = FastAPI()


# Pydantic model (request body)
class User(BaseModel):
    name: str
    age: int
    email: EmailStr
    tags: Optional[List[str]] = []


# API endpoint
@app.post("/create-user")
def create_user(user: User):
    return {
        "message": "User created successfully",
        "data": user
    }
