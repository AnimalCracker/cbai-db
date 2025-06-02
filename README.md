# Word Counting Dataset Generator

This project generates a dataset for testing large language models on word counting tasks. Each example in the dataset consists of a category type (e.g., fruit, animal, vehicle) and a list of words, where the task is to count how many words in the list belong to the given category.

## Example Format
```
Type: fruit
List: ['dog', 'apple', 'cherry', 'bus', 'cat', 'grape', 'bowl']
Answer: (3)
```

## Setup

1. Install uv if you haven't already:
```bash
pip install uv
```

2. Create and activate a virtual environment, then install dependencies:
```bash
uv venv
uv pip install -r requirements.txt
```

3. Run the generator:
```bash
python generate_dataset.py
```

This will:
- Download required NLTK data
- Generate 5000 examples using WordNet categories
- Save the dataset to `word_counting_dataset.json`
- Print a few random examples

## Dataset Structure

The generated JSON file contains a list of dictionaries, where each dictionary has:
- `type`: The category type (e.g., "fruit", "animal")
- `list`: List of words to count from
- `count`: The correct count (number of words matching the type)
- `matching_words`: List of words from the input that belong to the category
- `formatted_prompt`: The complete prompt in the desired format

## Categories

The dataset includes examples from various categories:
- Fruits
- Vegetables
- Animals
- Vehicles
- Furniture
- Tools
- Clothing
- Musical Instruments
- Sports
- Birds

Each example contains 4-11 words total, with 1-4 words from the target category.

## Model Evaluation

After generating the dataset, you can evaluate language models on the word counting task using the evaluation script:

```bash
python evaluate_models.py --model "meta-llama/Llama-2-7b-hf" --num_examples 100
```

Arguments:
- `--model`: HuggingFace model name/path (required)
- `--num_examples`: Number of examples to evaluate (default: 100)
- `--dataset`: Path to dataset file (default: word_counting_dataset.json)
- `--output`: Path to save results (default: evaluation_results.json)

The script will:
1. Load the specified model and tokenizer
2. Run zero-shot evaluation on the dataset
3. Save detailed results to a JSON file
4. Print summary metrics including:
   - Overall accuracy
   - Per-category accuracies

The evaluation is done zero-shot without any reasoning tokens or examples, testing the model's raw ability to count matching words.

### Suggested Models to Try
- meta-llama/Llama-2-7b-hf
- meta-llama/Llama-2-13b-hf
- mistralai/Mistral-7B-v0.1
- tiiuae/falcon-7b
- mosaicml/mpt-7b

Note: Some models may require authentication with the Hugging Face Hub. 