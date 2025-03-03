import time
import openai
from transformers import AutoTokenizer

MODEL_PATH="Qwen/Qwen2-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

port=62726
client = openai.Client(base_url=f"http://127.0.0.1:{port}/v1", api_key="None")

key=71432
n_repeat=5000
prompt = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. " * n_repeat + f"The pass key is {key}. Remember it. {key} is the pass key. " + "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. " * n_repeat + "What is the pass key?"

print("prompt length:", len(tokenizer.encode(prompt)))

start_time = time.time()
response = client.chat.completions.create(
    model=MODEL_PATH,
    messages=[
        {"role": "user", "content": prompt},
    ],
    temperature=0,
    stream=True,
)

first_token_received=False
for chunk in response:
    if chunk.choices[0].delta.content:
        if not first_token_received:
            print("\n\nTTFT:", time.time() - start_time)
            first_token_received=True
        print(chunk.choices[0].delta.content, end="", flush=True)
print("\n\nTime used:", time.time() - start_time)