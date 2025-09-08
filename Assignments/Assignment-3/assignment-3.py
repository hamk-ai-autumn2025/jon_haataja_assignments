import os
from openai import OpenAI
from dotenv import load_dotenv

# This text file was written with the help of ChatGPT (GPT-5)

load_dotenv()

# I'm using GitHub Models to access Openai API for "free" (there are usage limits of course)
# More info: https://github.blog/changelog/2025-05-19-github-models-built-into-your-repository-is-in-public-preview/
CUSTOM_BASE_URL = "https://models.github.ai/inference"
CUSTOM_API_KEY = os.getenv("MY_API_KEY")

def setup_client():
    client = OpenAI(
        base_url=CUSTOM_BASE_URL,
        api_key=CUSTOM_API_KEY
    )
    return client

def get_haiku_variations(client, topic):
    system_prompt = (
        "You are a creative haiku poet focused on SEO and rich language. "
        "Write a traditional 5-7-5 syllable haiku about the given topic. "
        "Use diverse, vivid synonyms and expressive, search-engine-friendly language. "
        "Avoid repetition. Focus on nature, emotion, and imagery. "
        "Respond only with the haiku, no explanations."
    )

    settings = [
        {"temperature": 0.7, "top_p": 0.9, "presence_penalty": 0.3, "frequency_penalty": 0.3},
        {"temperature": 0.9, "top_p": 1.0, "presence_penalty": 0.7, "frequency_penalty": 0.5},
        {"temperature": 0.6, "top_p": 0.8, "presence_penalty": 0.2, "frequency_penalty": 0.6},
    ]

    for i, params in enumerate(settings, 1):
        try:
            response = client.chat.completions.create(
                model="openai/gpt-4.1",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Write a creative, SEO-rich haiku about {topic} using vivid synonyms and beautiful imagery."}
                ],
                temperature=params["temperature"],
                top_p=params["top_p"],
                presence_penalty=params["presence_penalty"],
                frequency_penalty=params["frequency_penalty"],
                max_tokens=60,
            )
            haiku = response.choices[0].message.content.strip()
            print(f"\nHaiku {i} (T={params['temperature']}, P={params['top_p']}):")
            print(haiku)
            print("-" * 50)
        except Exception as e:
            print(f"Error generating Haiku {i}: {e}")

def main():
    print("🌸 Welcome to the SEO Haiku Generator! 🌸")
    topic = input("Enter a topic for your haiku: ").strip()
    if not topic:
        print("Topic cannot be empty. Exiting.")
        return

    try:
        client = setup_client()
        get_haiku_variations(client, topic)
    except Exception as e:
        print(f"Failed to connect to the LLM endpoint: {e}")
        print(f"Check if the server is running at {CUSTOM_BASE_URL}")
        exit(1)

    print("\n✨ Happy writing! ✨")

if __name__ == "__main__":
    main()