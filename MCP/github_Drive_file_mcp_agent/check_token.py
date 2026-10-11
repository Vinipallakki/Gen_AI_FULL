import json

with open("google_token.json", "r") as f:
    token = json.load(f)

print("\nToken information")
print("=" * 50)

print("Token type:", token.get("token_type"))

print("\nGranted scopes:")

print(
    token.get(
        "scope",
        "NO SCOPE FIELD FOUND"
    )
)