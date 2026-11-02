import os
import httpx
from dotenv import load_dotenv

load_dotenv()

token = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")

headers = {
    "Authorization": f"Bearer {token}",
    "Accept": "application/vnd.github+json"
}

response = httpx.get(
    "https://api.github.com/user",
    headers=headers
)

print("Status:", response.status_code)

if response.status_code == 200:
    print("GitHub Authentication Successful!")
    print("Username:", response.json()["login"])
else:
    print("Authentication Failed:", response.text)