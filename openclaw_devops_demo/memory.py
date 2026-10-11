memory_store = []


def save_incident(incident):
    memory_store.append(incident)


def get_last_incident(application):
    for incident in reversed(memory_store):
        if incident["application"] == application:
            return incident
    return None
