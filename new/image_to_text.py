import requests
import base64,json

def describe_image(image_path, model="llava:7b"):
    # Read and encode the image as base64
    with open(image_path, "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode("utf-8")

    # Send request to Ollama API
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": "Describe the objects in this image",
            "images": [img_base64]  # now properly base64 encoded
        },
        stream=True  # stream response line by line
    )

    # Collect response
    full_text = ""
    for line in response.iter_lines():
        if line:
            data = json.loads(line.decode("utf-8"))
            if "response" in data:
                full_text += data["response"]

    return full_text.strip()

if __name__ == "__main__":
    desciption = describe_image("image.png", model="llava:7b")
    print(desciption)
