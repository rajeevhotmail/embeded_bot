import requests
import time

API_URL = "http://127.0.0.1:5001/ask"

questions = [
    "Summarize the function definitions in the project",
    "How many Python files are in the project?",
    "Describe the purpose of this project",
    "What external libraries does this project use?"
]

for question in questions:
    payload = {"question": question}

    start_time = time.time()
    response = requests.post(API_URL, json=payload)
    total_time = time.time() - start_time

    try:
        result = response.json()
        print(f"🔹 Query: {question}")
        print(f"⏳ Response Time: {total_time:.2f} seconds")
        print(f"📜 Answer: {result.get('result', {}).get('answer', 'No answer')}")
        print("-" * 50)
    except Exception as e:
        print(f"❌ Error processing request: {str(e)}")
