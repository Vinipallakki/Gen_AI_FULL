def check_application_status():
    return {
        "application": "payment-api",
        "status": "DOWN"
    }


def get_application_logs():
    return {
        "logs": [
            "ERROR: Database connection timeout",
            "ERROR: PostgreSQL connection failed"
        ]
    }
