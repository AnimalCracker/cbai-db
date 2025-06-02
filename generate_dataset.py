import nltk
from nltk.corpus import wordnet as wn
import random
import json
from typing import List, Tuple, Dict
import tqdm

def download_nltk_data():
    """Download required NLTK data."""
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('words')

def get_category_words() -> Dict[str, List[str]]:
    """Get words for different categories from WordNet."""
    categories = {
        'fruit': wn.synset('fruit.n.01'),
        'vegetable': wn.synset('vegetable.n.01'),
        'animal': wn.synset('animal.n.01'),
        'vehicle': wn.synset('vehicle.n.01'),
        'furniture': wn.synset('furniture.n.01'),
        'tool': wn.synset('tool.n.01'),
        'clothing': wn.synset('clothing.n.01'),
        'instrument': wn.synset('musical_instrument.n.01'),
        'sport': wn.synset('sport.n.01'),
        'bird': wn.synset('bird.n.01'),
    }
    
    category_words = {}
    for category_name, synset in categories.items():
        # Get all hyponyms
        hyponyms = synset.hyponyms()
        words = set()
        for hyp in hyponyms:
            # Get lemma names and filter out multi-word expressions
            words.update([lemma.name() for lemma in hyp.lemmas() if '_' not in lemma.name()])
        category_words[category_name] = list(words)
    
    return category_words

def generate_example(category: str, category_words: List[str], other_words: List[str]) -> Tuple[str, List[str], int, List[str]]:
    """Generate a single example."""
    # Randomly select how many category words to include (1-4)
    n_category_words = random.randint(1, 4)
    
    # Select random words from the category
    selected_category_words = random.sample(category_words, n_category_words)
    
    # Select random other words
    n_other_words = random.randint(3, 7)  # Total list length will be 4-11 words
    selected_other_words = random.sample(other_words, n_other_words)
    
    # Combine and shuffle
    all_words = selected_category_words + selected_other_words
    random.shuffle(all_words)
    
    return category, all_words, n_category_words, selected_category_words

def create_dataset(n_examples: int = 5000) -> List[Dict]:
    """Create the full dataset."""
    # Get category words
    category_words = get_category_words()
    
    # Create a pool of "other" words that don't belong to any category
    brown_words = set(word.lower() for word in nltk.corpus.words.words())
    other_words = list(brown_words - set().union(*category_words.values()))
    
    dataset = []
    for _ in tqdm.tqdm(range(n_examples)):
        # Randomly select a category
        category = random.choice(list(category_words.keys()))
        
        # Generate example
        cat, word_list, count, matching_words = generate_example(
            category,
            category_words[category],
            other_words
        )
        
        # Format example
        example = {
            'type': cat,
            'list': word_list,
            'count': count,
            'matching_words': matching_words,
            'formatted_prompt': f'Type: {cat}\nList: {word_list}\nAnswer: (',  # Only include open parenthesis
            'full_prompt': f'Type: {cat}\nList: {word_list}\nAnswer: ({count})'  # Keep full prompt for reference
        }
        dataset.append(example)
    
    return dataset

def main():
    # Download required NLTK data
    download_nltk_data()
    
    # Generate dataset
    print("Generating dataset...")
    dataset = create_dataset(5000)
    
    # Save to JSON file
    with open('word_counting_dataset.json', 'w') as f:
        json.dump(dataset, f, indent=2)
    
    # Print a few examples
    print("\nExample prompts:")
    for example in random.sample(dataset, 3):
        print("\n" + example['formatted_prompt'])

if __name__ == "__main__":
    main() 