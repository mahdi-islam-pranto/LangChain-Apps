from transformers import pipeline

messages = [
    {"role": "user", "content": "write a paragraph about Bangladesh"},
]

pipe = pipeline(
    "text-generation",
    model="facebook/opt-125m",  # Much smaller model (125M parameters)
    device=-1  # Force CPU usage
)

# Generate a response
response = pipe(messages[0]["content"])
print(response[0]["generated_text"])
