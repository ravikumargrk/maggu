import requests

# test connection
text = requests.get('http://localhost:11434/').text

if not (text=='Ollama is running'):
    print('Error: Ollama is not running.')
    exit(0)

import sys 
tokens = sys.argv[1:]

import json
payload = {
    "model": "mistral",
    "messages": [
        {
            "role": "user",
            "content": ' '.join(tokens)
        }
    ]
}

chat_endpoint = r'http://localhost:11434/api/chat'

with requests.post(url=chat_endpoint, json=payload, stream=True) as response:
    response.raise_for_status()
    for chunk in response.iter_lines(decode_unicode=False):
        if chunk:  # Filter out keep-alive chunks
            if isinstance(chunk, bytes):
                line_json = json.loads(chunk.decode())
                print(line_json['message']['content'], end='', flush=True)

