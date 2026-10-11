import asyncio
import json
import os
import webbrowser

from urllib.parse import urlencode
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

import requests


# ============================================================
# CONFIGURATION
# ============================================================

GOOGLE_CLIENT_FILE = "google_client_secret.json"

REDIRECT_URI = "http://localhost:8080/oauth2callback"

MCP_URL = "https://drivemcp.googleapis.com/mcp/v1"

SCOPES = [
    "https://www.googleapis.com/auth/drive.readonly",
    "https://www.googleapis.com/auth/drive.file",
]


# ============================================================
# LOAD GOOGLE CLIENT CREDENTIALS
# ============================================================

with open(
    GOOGLE_CLIENT_FILE,
    "r"
) as file:

    credentials = json.load(file)


# Google web application credentials
web_credentials = credentials["web"]

CLIENT_ID = web_credentials["client_id"]

CLIENT_SECRET = web_credentials["client_secret"]


# ============================================================
# CALLBACK STORAGE
# ============================================================

authorization_code = None


class OAuthHandler(
    BaseHTTPRequestHandler
):

    def do_GET(self):

        global authorization_code

        # Parse query parameters
        from urllib.parse import urlparse, parse_qs

        parsed_url = urlparse(
            self.path
        )

        params = parse_qs(
            parsed_url.query
        )

        if "code" in params:

            authorization_code = (
                params["code"][0]
            )

            self.send_response(200)

            self.send_header(
                "Content-Type",
                "text/html"
            )

            self.end_headers()

            self.wfile.write(
                b"""
                <html>
                <body>
                <h2>Google authentication successful.</h2>
                <p>You can close this browser window.</p>
                </body>
                </html>
                """
            )

        else:

            self.send_response(400)

            self.end_headers()

            self.wfile.write(
                b"Authorization failed."
            )


# ============================================================
# START CALLBACK SERVER
# ============================================================

def start_callback_server():

    server = HTTPServer(
        (
            "localhost",
            8080
        ),
        OAuthHandler
    )

    thread = Thread(
        target=server.handle_request,
        daemon=True
    )

    thread.start()

    return server


# ============================================================
# GOOGLE OAUTH
# ============================================================

def authenticate_google():

    global authorization_code

    authorization_code = None

    # --------------------------------------------------------
    # Authorization URL
    # --------------------------------------------------------

    params = {

        "client_id":
            CLIENT_ID,

        "redirect_uri":
            REDIRECT_URI,

        "response_type":
            "code",

        "scope":
            " ".join(SCOPES),

        "access_type":
            "offline",

        "prompt":
            "consent"
    }


    authorization_url = (

        "https://accounts.google.com/o/oauth2/v2/auth?"

        + urlencode(params)

    )


    print(
        "\nOpening Google authentication..."
    )


    # --------------------------------------------------------
    # Start callback server
    # --------------------------------------------------------

    server = (
        start_callback_server()
    )


    # --------------------------------------------------------
    # Open browser
    # --------------------------------------------------------

    webbrowser.open(
        authorization_url
    )


    print(
        "\nPlease login to Google "
        "and authorize Drive access."
    )


    # --------------------------------------------------------
    # Wait for callback
    # --------------------------------------------------------

    while authorization_code is None:

        import time

        time.sleep(0.5)


    server.server_close()


    print(
        "\nAuthorization code received."
    )


    # ========================================================
    # EXCHANGE CODE FOR TOKEN
    # ========================================================

    token_response = requests.post(

        "https://oauth2.googleapis.com/token",

        data={

            "code":
                authorization_code,

            "client_id":
                CLIENT_ID,

            "client_secret":
                CLIENT_SECRET,

            "redirect_uri":
                REDIRECT_URI,

            "grant_type":
                "authorization_code"
        },

        timeout=30
    )


    token_response.raise_for_status()


    token = token_response.json()


    # --------------------------------------------------------
    # Save token
    # --------------------------------------------------------

    with open(
        "google_token.json",
        "w"
    ) as file:

        json.dump(
            token,
            file,
            indent=2
        )


    print(
        "Google token saved."
    )


    return token


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 60
    )

    print(
        "GOOGLE DRIVE MCP AUTH TEST"
    )

    print(
        "=" * 60
    )


    token = authenticate_google()


    print(
        "\nAuthentication completed."
    )


    print(
        "\nAccess token received:"
    )

    print(
        token.get("access_token")[:20]
        + "..."
    )


if __name__ == "__main__":

    main()