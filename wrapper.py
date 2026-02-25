import os
import argparse
from groq import Groq
from dotenv import load_dotenv

#Initialize 
load_dotenv()
print("API KEY:", os.getenv("GROQ_API_KEY"))
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def run_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", help="Your question for the LLM")
    parser.add_argument("--temp", type=float, default=0.7)
    args = parser.parse_args()
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=args.temp,
        messages=[
            {"role": "user", "content": args.prompt}
        ]
    )
    print(response.choices[0].message.content)


run_cli()