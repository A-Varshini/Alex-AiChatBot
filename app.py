from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

app = Flask(__name__)
CORS(app)  # Allow frontend to connect

# Load BlenderBot model
MODEL_NAME = "facebook/blenderbot-400M-distill"
tokenizer = BlenderbotTokenizer.from_pretrained(MODEL_NAME)
model = BlenderbotForConditionalGeneration.from_pretrained(MODEL_NAME)

# Function to generate chatbot response
def chatbot_response(user_input):
    inputs = tokenizer(user_input, return_tensors="pt")
    reply_ids = model.generate(**inputs)
    return tokenizer.decode(reply_ids[0], skip_special_tokens=True)

# API route for chatbot messages
@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message")
    bot_reply = chatbot_response(user_message)
    return jsonify({"response": bot_reply})

if __name__ == "__main__":
    app.run(debug=True)
