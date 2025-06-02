from extract_examples import extract_example_pairs, load_results

# Load GPT-2 medium results
results = load_results('gpt2_medium_1k_results.json')

# Extract example pairs
pairs = extract_example_pairs(results, num_pairs=50)

# Save pairs
with open('gpt2_medium_example_pairs.json', 'w') as f:
    import json
    json.dump(pairs, f, indent=2)

print(f'Extracted {len(pairs)} pairs from GPT-2 medium results')

# Print a sample pair
if pairs:
    print("\nSample pair:")
    correct, incorrect = pairs[0]
    print("\nCorrect example:")
    print(correct)
    print("\nIncorrect example:")
    print(incorrect) 