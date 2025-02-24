import os
from flask import Flask, render_template, request, jsonify
from gpt4all import GPT4All

os.environ["GPT4ALL_NO_CUDA"] = "1"

app = Flask(__name__)


# Set model path to the same directory as app.py
model_path = os.path.join(os.path.dirname(__file__), "Phi-3-mini-4k-instruct.Q4_0.gguf")

# Load the model from the local file
model = GPT4All(model_path, allow_download=False)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message")
    response = model.generate(user_message)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
