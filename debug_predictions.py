import json
from transformer_lens import HookedTransformer
import torch

# Load model
print("Loading model...")
model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")

# Load one example pair
with open("gpt2_example_pairs.json", 'r') as f:
    example_pairs = json.load(f)

correct_ex, incorrect_ex = example_pairs[0]

print("Correct example:")
print(correct_ex)
print("\nIncorrect example:")
print(incorrect_ex)

def debug_number_prediction(model, prompt):
    """Debug version of get_number_prediction."""
    tokens = model.to_tokens(prompt)[0]
    
    # Find answer position - look for the number after "Answer: ("
    paren_pos = prompt.find("Answer: (") + 9  # Position of number
    answer_position = len(model.to_tokens(prompt[:paren_pos])[0])  # Don't subtract 1
    
    print(f"\nPrompt slice: '{prompt[:paren_pos]}'")
    print(f"Answer position: {answer_position}")
    if answer_position < len(tokens):
        print(f"Token at answer position: {tokens[answer_position]} -> '{model.to_string(tokens[answer_position])}'")
    else:
        print("Answer position is out of bounds!")
        return None
    
    # Get logits
    with torch.no_grad():
        logits = model(prompt)
    
    print(f"Logits shape: {logits.shape}")
    
    # Check number token predictions - fix the tokenization
    number_tokens = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    number_token_ids = []
    for num in number_tokens:
        token_id = model.to_tokens(num)[0][-1]  # Get the last token
        number_token_ids.append(token_id)
    
    number_token_ids = torch.tensor(number_token_ids)
    print(f"Number token IDs: {number_token_ids}")
    
    answer_logits = logits[0, answer_position]
    print(f"Answer logits shape: {answer_logits.shape}")
    
    number_logits = answer_logits[number_token_ids]
    print(f"Number logits: {number_logits}")
    
    predicted_idx = number_logits.argmax()
    predicted_token_id = number_token_ids[predicted_idx]
    predicted_number = int(model.to_string(predicted_token_id))
    
    print(f"Predicted number: {predicted_number}")
    
    return predicted_number

print("\n" + "="*50)
print("Debugging correct example:")
pred1 = debug_number_prediction(model, correct_ex)

print("\n" + "="*50)
print("Debugging incorrect example:")
pred2 = debug_number_prediction(model, incorrect_ex)

print(f"\nPrediction difference: {abs(pred1 - pred2)}") 