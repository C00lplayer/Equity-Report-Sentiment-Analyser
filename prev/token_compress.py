import tiktoken
import openai

enc = tiktoken.encoding_for_model("gpt-4o-mini")   # or the model you're using
MAX_TOKENS = 450

def compress_to_limit(text):
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You compress financial text while preserving sentiment."},
            {"role": "user", "content": text}
        ]
    )
    output = response.choices[0].message.content

    # Count tokens
    tokens = enc.encode(output)
    
    # If over limit → automatically re-compress
    while len(tokens) > MAX_TOKENS:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": 
                    f"Reduce further while preserving sentiment. Output must remove neutral details. "
                    f"Current token count is {len(tokens)} and must be <= {MAX_TOKENS}."
                },
                {"role": "user", "content": output}
            ]
        )
        output = response.choices[0].message.content
        tokens = enc.encode(output)

    return output
