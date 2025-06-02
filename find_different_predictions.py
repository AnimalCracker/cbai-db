import json
from transformer_lens import HookedTransformer
import torch
from tqdm import tqdm

def get_model_prediction(model, prompt):
    """Get the model's prediction for a counting prompt."""
    # Find answer position
    answer_text_pos = prompt.find("Answer: (") + 9
    answer_position = len(model.to_tokens(prompt[:answer_text_pos])[0])
    
    # Get logits
    with torch.no_grad():
        logits = model(prompt)
    
    # Get number prediction
    number_tokens = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    number_token_ids = []
    for num in number_tokens:
        token_id = model.to_tokens(num)[0][-1]
        number_token_ids.append(token_id)
    
    number_token_ids = torch.tensor(number_token_ids)
    number_logits = logits[0, answer_position][number_token_ids]
    predicted_idx = number_logits.argmax()
    return int(number_tokens[predicted_idx])

def main():
    print("Loading model...")
    model = HookedTransformer.from_pretrained("gpt2-small", device="cuda")
    
    # Load all example pairs
    with open("gpt2_example_pairs.json", 'r') as f:
        example_pairs = json.load(f)
    
    print(f"Checking {len(example_pairs)} example pairs for different predictions...")
    
    good_pairs = []
    
    for i, (correct_ex, incorrect_ex) in enumerate(tqdm(example_pairs)):
        try:
            pred1 = get_model_prediction(model, correct_ex)
            pred2 = get_model_prediction(model, incorrect_ex)
            
            if pred1 != pred2:
                print(f"\nFound different predictions at pair {i}:")
                print(f"Example A predicts: {pred1}")
                print(f"Example B predicts: {pred2}")
                good_pairs.append((correct_ex, incorrect_ex))
                
                if len(good_pairs) >= 10:  # Stop after finding 10 good pairs
                    break
                    
        except Exception as e:
            print(f"Error processing pair {i}: {e}")
            continue
    
    print(f"\nFound {len(good_pairs)} pairs with different predictions")
    
    if good_pairs:
        # Save the good pairs
        with open("gpt2_different_prediction_pairs.json", 'w') as f:
            json.dump(good_pairs, f, indent=2)
        print("Saved to gpt2_different_prediction_pairs.json")
    else:
        print("No pairs with different predictions found!")

if __name__ == "__main__":
    main() 