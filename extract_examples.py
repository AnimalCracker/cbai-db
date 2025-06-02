import json
from typing import List, Tuple, Dict
import random

def load_results(filepath: str) -> Dict:
    """Load evaluation results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def extract_example_pairs(results: Dict, 
                        num_pairs: int = 50,
                        min_list_length: int = 5,
                        max_list_length: int = 10) -> List[Tuple[str, str]]:
    """
    Extract pairs of correct and incorrect counting examples.
    Each pair consists of examples with the same category and similar list length,
    where one got the correct count and one got it wrong.
    
    Args:
        results: Dictionary containing evaluation results
        num_pairs: Number of example pairs to extract
        min_list_length: Minimum length of word lists to consider
        max_list_length: Maximum length of word lists to consider
    
    Returns:
        List of (correct_example, incorrect_example) tuples
    """
    # Get all detailed results within list length constraints
    detailed_results = results["detailed_results"]
    filtered_results = [
        r for r in detailed_results
        if min_list_length <= r["list_length"] <= max_list_length
    ]
    
    # Group by category
    category_groups = {}
    for result in filtered_results:
        category = result["category"]
        if category not in category_groups:
            category_groups[category] = {"correct": [], "incorrect": []}
            
        if result["is_correct"]:
            category_groups[category]["correct"].append(result)
        else:
            category_groups[category]["incorrect"].append(result)
    
    # Find categories that have both correct and incorrect examples
    valid_categories = [
        cat for cat, group in category_groups.items()
        if group["correct"] and group["incorrect"]
    ]
    
    if not valid_categories:
        print("No valid example pairs found!")
        return []
    
    # Extract pairs
    pairs = []
    pairs_per_category = max(1, num_pairs // len(valid_categories))
    
    for category in valid_categories:
        group = category_groups[category]
        
        # Get pairs for this category
        category_pairs = []
        correct_examples = group["correct"]
        incorrect_examples = group["incorrect"]
        
        # Try to match examples with similar list lengths
        for correct in correct_examples:
            correct_len = correct["list_length"]
            # Find incorrect examples with similar length (±1)
            matching_incorrect = [
                inc for inc in incorrect_examples
                if abs(inc["list_length"] - correct_len) <= 1
            ]
            
            if matching_incorrect:
                incorrect = random.choice(matching_incorrect)
                category_pairs.append((
                    correct["prompt"] + str(correct["correct_count"]) + ")",
                    incorrect["prompt"] + str(incorrect["predicted_count"]) + ")"
                ))
                
                if len(category_pairs) >= pairs_per_category:
                    break
        
        pairs.extend(category_pairs)
        
        if len(pairs) >= num_pairs:
            break
    
    # Shuffle and limit to requested number
    random.shuffle(pairs)
    return pairs[:num_pairs]

def main():
    # Load results for each model
    models = {
        "gpt2": "gpt2_2k_results.json",
        "gemma": "gemma_2k_results.json",
        "phi2": "phi2_2k_results.json"
    }
    
    for model_name, results_file in models.items():
        print(f"\nExtracting examples for {model_name}...")
        
        try:
            # Load results
            results = load_results(results_file)
            
            # Extract example pairs
            pairs = extract_example_pairs(results)
            
            if not pairs:
                print(f"No valid example pairs found for {model_name}")
                continue
                
            # Save pairs
            output_file = f"{model_name}_example_pairs.json"
            with open(output_file, 'w') as f:
                json.dump(pairs, f, indent=2)
                
            print(f"Saved {len(pairs)} example pairs to {output_file}")
            
            # Print a sample pair
            print("\nSample pair:")
            correct, incorrect = pairs[0]
            print("\nCorrect example:")
            print(correct)
            print("\nIncorrect example:")
            print(incorrect)
            
        except Exception as e:
            print(f"Error processing {model_name}: {str(e)}")

if __name__ == "__main__":
    main() 