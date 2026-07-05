import requests
import json

BASE_URL = "http://localhost:8000"

TEST_PROMPTS = [
    "Check the status of order #12345.",
    "Find the refund policy and email it to the customer.",
    "Update customer phone number after verifying identity."
]

print("=" * 60)
print("PHASE 5 - GROQ LLM INTEGRATION VERIFICATION")
print("=" * 60)

for i, prompt in enumerate(TEST_PROMPTS, start=1):

    print(f"\nTest {i}")
    print("-" * 60)
    print("Prompt:", prompt)

    response = requests.post(
        f"{BASE_URL}/simulate",
        json={"prompt": prompt}
    )

    if response.status_code != 200:
        print("[X] API FAILED")
        print(response.text)
        continue

    result = response.json()

    print("\nSession ID:")
    print(result.get("session_id"))

    print("\nLLM Provider:")
    print(result.get("provider"))

    print("\nModel:")
    print(result.get("model"))

    print("\nTools Selected:")

    tools = result.get("tool_sequence", [])

    if not tools:
        print("[X] No tools returned")
    else:
        for t in tools:
            print("  -", t)

    print("\nTrace Length:", len(tools))

    print("\nAgent Response:")
    print(result.get("response"))

print("\n" + "=" * 60)
print("PHASE 5 VERIFICATION COMPLETE")
print("=" * 60)