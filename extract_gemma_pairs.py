from extract_examples import extract_example_pairs, load_results

# Load Gemma-2B results  
results = load_results('gemma_2b_1k_results.json')

# Extract example pairs
pairs = extract_example_pairs(results, num_pairs=50)

# Save pairs
with open('gemma_2b_example_pairs.json', 'w') as f:
    import json
    json.dump(pairs, f, indent=2)

print(f'Extracted {len(pairs)} pairs from Gemma-2B results')

# Print a sample pair
if pairs:
    print("\nSample pair:")
    correct, incorrect = pairs[0]
    print("\nCorrect example:")
    print(correct)
    print("\nIncorrect example:")
    print(incorrect) 