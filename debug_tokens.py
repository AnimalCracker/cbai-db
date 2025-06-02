import json
from transformer_lens import HookedTransformer

# Load model
model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")

# Load one example
with open("gpt2_example_pairs.json", 'r') as f:
    example_pairs = json.load(f)

correct_ex, incorrect_ex = example_pairs[0]

print("Correct example:")
print(correct_ex)
print("\nTokens:")
tokens = model.to_tokens(correct_ex)
print(tokens)
print("\nDecoded tokens:")
for i, token in enumerate(tokens[0]):
    print(f"{i}: {token.item()} -> '{model.to_string(token)}'")

print("\n" + "="*50)
print("\nIncorrect example:")
print(incorrect_ex)
print("\nTokens:")
tokens = model.to_tokens(incorrect_ex)
print(tokens)
print("\nDecoded tokens:")
for i, token in enumerate(tokens[0]):
    print(f"{i}: {token.item()} -> '{model.to_string(token)}'") 