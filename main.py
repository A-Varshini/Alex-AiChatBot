from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

# Load the model and tokenizer
MODEL_NAME = "facebook/blenderbot-400M-distill"
tokenizer = BlenderbotTokenizer.from_pretrained(MODEL_NAME)
model = BlenderbotForConditionalGeneration.from_pretrained(MODEL_NAME)

def chatbot_response(user_input):
    inputs = tokenizer(user_input, return_tensors="pt")
    reply_ids = model.generate(**inputs)
    return tokenizer.decode(reply_ids[0], skip_special_tokens=True)

if __name__ == "__main__":
    print("Chatbot is running! Type 'exit' to stop.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Chatbot: Goodbye!")
            break
        response = chatbot_response(user_input)
        print("Chatbot:", response)
