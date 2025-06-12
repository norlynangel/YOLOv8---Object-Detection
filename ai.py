from openai import OpenAIError

OPENAI_API_KEY = "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"
client = OpenAI(api_key=OPENAI_API_KEY)
completion = client.chat.completions.create(
    model="gpt-3.5o-mini"
    messages=[
        {"role": "user", "content": "Tell me about the detected object."}
    ]
)
print (completion.choices[0].message.content)