import requests

BASE_URL = "http://127.0.0.1:5000/posts"

def test_get():
    response = requests.get(BASE_URL)
    print("GET:", response.status_code, response.json())

def test_post():
    response = requests.post(BASE_URL, json={"title": "test"})
    print("POST:", response.status_code, response.json())

def test_put():
    response = requests.put(f"{BASE_URL}/1", json={"title": "updated"})
    print("PUT:", response.status_code, response.json())

def test_delete():
    response = requests.delete(f"{BASE_URL}/1")
    print("DELETE:", response.status_code, response.json())

if __name__ == "__main__":
    test_get()
    # test_post()
    # test_put()
    # test_delete()