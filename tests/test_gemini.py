from google import genai
from google.genai.types import HttpOptions

def get_gemini_response(model: str, prompt: str) -> str:
    # client automatically picks up GEMINI_API_KEY and GOOGLE_GENAI_USE_VERTEXAI
    client = genai.Client(http_options=HttpOptions(api_version="v1"))
   
    response = client.models.generate_content(
 model=model,
 contents=prompt,
    )
    return response.text

if __name__ == "__main__":
    prompt = "What is the capital of Japan?"
    response = get_gemini_response(model="gemini-2.5-flash", prompt=prompt)
    print(response)