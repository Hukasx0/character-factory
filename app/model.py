import requests
import json

# The URL to your local server
url = 'http://localhost:11434/api/generate'

headers = {
    'Content-Type': 'application/json',
}

def send_message(prompt, context=None):
    # The data payload for POST request
    data = {
        "model": "llama2-uncensored",  # Use the correct model name
        "prompt": prompt,
        "stream": False,  # Set stream to False to get the complete response at once
    }

    if context:
        data['context'] = context  # Include context if available

    # Send the POST request to the server
    response = requests.post(url, data=json.dumps(data), headers=headers)

    # Check if the response is successful
    if response.status_code == 200:
        json_response = response.json()
        response_text = json_response.get('response', '').replace('\\n', '\n').strip()
        return response_text, json_response.get('context')  # Return response and context
    else:
        print(f"Error: {response.status_code}")
        return None, None

def chat():
    print("You're now chatting with the model. Type 'quit' to exit.")
    context = None  # Initialize context
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
        model_response, context = send_message(user_input, context)
        if model_response:
            print(f"Model: {model_response}")

chat()
