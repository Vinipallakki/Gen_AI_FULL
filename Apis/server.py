from flask import Flask, request, jsonify

app = Flask(__name__)

# Fake database
data_store = [
    {"id": 1, "title": "First post"}
]

# 🔹 GET
@app.route('/posts', methods=['GET'])
def get_posts():
    return jsonify(data_store)

# 🔹 POST
@app.route('/posts', methods=['POST'])
def create_post():
    new_data = request.json
    new_data["id"] = len(data_store) + 1
    data_store.append(new_data)
    return jsonify(new_data), 201

# 🔹 PUT
@app.route('/posts/<int:id>', methods=['PUT'])
def update_post(id):
    for item in data_store:
        if item["id"] == id:
            item["title"] = request.json.get("title")
            return jsonify(item)
    return {"error": "Not found"}, 404

# 🔹 DELETE
@app.route('/posts/<int:id>', methods=['DELETE'])
def delete_post(id):
    global data_store
    data_store = [item for item in data_store if item["id"] != id]
    return {"message": "Deleted"}

if __name__ == "__main__":
    app.run(debug=True)