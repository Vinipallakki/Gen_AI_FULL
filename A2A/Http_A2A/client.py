import requests

city = input("Enter City: ")

response = requests.post(
    "http://localhost:8000/invoke",
    json={
        "city": city
    }
)

print(response.json())