import openai
import os

# Set your OpenAI API key as an environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set")

def generate_response(prompt: str) -> str:
    """
    Generate a response using OpenAI's GPT-3 API.

    Args:
        prompt: The input prompt for GPT-3

    Returns:
        The generated text response

    Raises:
        ValueError: If prompt is empty
        openai.error.OpenAIError: If the API call fails
    """
    if not prompt:
        raise ValueError("Prompt cannot be empty")

    try:
        response = openai.Completion.create(
            engine="davinci",
            prompt=prompt,
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.7,
        )
        return response.choices[0].text.strip()
    except openai.error.OpenAIError as e:
        raise RuntimeError(f"OpenAI API error: {str(e)}") from e
