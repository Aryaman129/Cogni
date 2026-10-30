import os
import sys
from openai import OpenAI

api_key = os.environ.get("NEBIUS_API_KEY")
base_url = os.environ.get("NEBIUS_BASE_URL", "https://api.tokenfactory.eu-north1.nebius.com/v1/")
model = os.environ.get("NEBIUS_MODEL_ID", "nvidia/Nemotron-3_5-Lightning")

if not api_key:
    print("NEBIUS_API_KEY not found in environment")
    sys.exit(1)

client = OpenAI(
    base_url=base_url,
    api_key=api_key,
)

try:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are CogniSafe's clinical explanation assistant."},
            {"role": "user", "content": "Return a one sentence observation that speech rate is stable."}
        ],
        max_tokens=60
    )
    print("LIVE INFERENCE SUCCESSFUL!")
    print(f"Model used: {response.model}")
    print(f"Response: {response.choices[0].message.content}")
except Exception as exc:
    print(f"Inference error: {type(exc).__name__}: {exc}")
