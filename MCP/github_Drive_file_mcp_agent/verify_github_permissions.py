
import os
import httpx
from dotenv import load_dotenv

# Load the latest .env value
load_dotenv(override=True)

TOKEN = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")

if not TOKEN:
    raise ValueError("GitHub token missing")

headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28"
}

BASE_URL = "https://api.github.com"

with httpx.Client(headers=headers, timeout=30) as client:

    # 1. Verify authenticated account
    user_response = client.get(f"{BASE_URL}/user")

    print("\n========== AUTHENTICATION ==========")
    print("Status:", user_response.status_code)

    if user_response.status_code != 200:
        print(user_response.text)
        raise SystemExit("GitHub authentication failed")

    print("Authenticated user:", user_response.json()["login"])

    # 2. Check repository access
    repo_response = client.get(
        f"{BASE_URL}/repos/vinipallakki/Gen_AI_FULL"
    )

    print("\n========== REPOSITORY ACCESS ==========")
    print("Status:", repo_response.status_code)

    if repo_response.status_code != 200:
        print(repo_response.text)
        raise SystemExit("Cannot access repository")

    repo = repo_response.json()

    print("Repository:", repo["full_name"])
    print("Private:", repo["private"])
    print("Default branch:", repo["default_branch"])

    permissions = repo.get("permissions", {})

    print("\nEffective repository permissions:")

    for permission, value in permissions.items():
        print(f"{permission}: {value}")

    # 3. Check branch write permission
    print("\n========== WRITE ACCESS ==========")

    print(
        "Can push:",
        permissions.get("push", "Not returned")
    )

    print(
        "Admin:",
        permissions.get("admin", "Not returned")
    )

    print(
        "Maintain:",
        permissions.get("maintain", "Not returned")
    )

    print("\n========== TOKEN PERMISSION HEADERS ==========")

    print(
        "Accepted permissions:",
        repo_response.headers.get(
            "X-Accepted-GitHub-Permissions",
            "Not provided"
        )
    )

    print("\nDiagnostic completed.")
