from tools import check_application_status, get_application_logs
from memory import save_incident, get_last_incident


def investigate_application(application="payment-api"):
    print("\n[AGENT] Checking application status...")
    status = check_application_status()

    if status["status"] == "DOWN":
        print("[AGENT] Application is DOWN. Calling log tool...")
        log_result = get_application_logs()

        incident = {
            "application": application,
            "status": status["status"],
            "logs": log_result["logs"],
            "root_cause": "Possible PostgreSQL database connection timeout"
        }

        save_incident(incident)

        return (
            f"Application: {application}\n"
            f"Status: {status['status']}\n"
            f"Possible root cause: PostgreSQL database connection timeout.\n"
            f"Incident stored in memory."
        )

    return f"Application: {application}\nStatus: HEALTHY"


def previous_issue(application="payment-api"):
    incident = get_last_incident(application)

    if not incident:
        return "No previous incident found."

    return (
        f"Previous incident for {application}:\n"
        f"Status: {incident['status']}\n"
        f"Root cause: {incident['root_cause']}"
    )
