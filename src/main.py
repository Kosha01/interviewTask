from flask import Flask, render_template, request, jsonify
from llm_provider import LLMProvider

app = Flask(__name__)
llm = LLMProvider()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")
    response = llm.generate(user_message)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)

