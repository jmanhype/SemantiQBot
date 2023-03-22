import openai
import os

openai.api_key = os.getenv("your_api_key")
  # set your OpenAI API key as an environment variable

def generate_response(prompt):
    response = openai.Completion.create(
        engine="davinci",
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.7,
    )

    return response.choices[0].text.strip()
