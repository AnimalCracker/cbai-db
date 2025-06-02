import json
from transformer_lens import HookedTransformer
import torch
from tqdm import tqdm

def get_model_prediction(model, prompt):
    """Get the model's prediction for a counting prompt."""
    try:
        # Get logits
        with torch.no_grad():
            logits = model(prompt)
        
        # Use the last position as answer position (where the model would generate next)
        answer_position = logits.shape[1] - 1
        
        # Get number prediction
        number_tokens = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        number_token_ids = []
        for num in number_tokens:
            token_id = model.to_tokens(num)[0][-1]
            number_token_ids.append(token_id)
        
        number_token_ids = torch.tensor(number_token_ids).to(model.cfg.device)
        number_logits = logits[0, answer_position][number_token_ids]
        predicted_idx = number_logits.argmax()
        return int(number_tokens[predicted_idx])
    except Exception as e:
        print(f"Error in get_model_prediction: {e}")
        return None

def main():
    print("Loading Gemma-2-2B with TransformerLens...")
    model = HookedTransformer.from_pretrained("gemma-2-2b", device="cuda")
    
    # Create some artificial pairs from the evaluation results
    # Load evaluation results to get examples
    with open("gemma_2b_1k_results.json", 'r') as f:
        results = json.load(f)
    
    # Get examples from different categories that might predict differently
    examples = []
    for result in results["detailed_results"][:20]:  # Take first 20 examples
        examples.append(result["prompt"])
    
    print(f"Checking {len(examples)} examples for different predictions...")
    
    good_pairs = []
    
    # Compare all examples pairwise to find ones with different predictions
    for i in tqdm(range(len(examples))):
        for j in range(i+1, len(examples)):
            try:
                pred1 = get_model_prediction(model, examples[i])
                pred2 = get_model_prediction(model, examples[j])
                
                # Skip if either prediction failed
                if pred1 is None or pred2 is None:
                    continue
                
                if pred1 != pred2:
                    print(f"\nFound different predictions between examples {i} and {j}:")
                    print(f"Example {i} predicts: {pred1}")
                    print(f"Example {j} predicts: {pred2}")
                    good_pairs.append((examples[i], examples[j]))
                    
                    if len(good_pairs) >= 10:  # Stop after finding 10 good pairs
                        break
                        
            except Exception as e:
                print(f"Error processing pair {i},{j}: {e}")
                continue
        
        if len(good_pairs) >= 10:
            break
    
    print(f"\nFound {len(good_pairs)} pairs with different predictions")
    
    if good_pairs:
        # Save the good pairs
        with open("gemma_2b_different_prediction_pairs.json", 'w') as f:
            json.dump(good_pairs, f, indent=2)
        print("Saved to gemma_2b_different_prediction_pairs.json")
    else:
        print("No pairs with different predictions found!")

if __name__ == "__main__":
    main() 