import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict
import numpy as np
from tqdm import tqdm
import re
import argparse

class ModelEvaluator:
    def __init__(self, model_name: str, device: str = "cuda"):
        """Initialize model and tokenizer."""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Please check your GPU setup.")
        
        self.model_name = model_name
        self.device = device
        print(f"Loading {model_name}...")
        print(f"Using device: {device}")
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device,
            load_in_8bit=True
        )
        
        # Clear CUDA cache after model loading
        torch.cuda.empty_cache()
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        
    def extract_answer(self, response: str) -> int:
        """Extract the numerical answer from the model's response."""
        # Look for a number in parentheses
        match = re.search(r'\((\d+)\)', response)
        if match:
            return int(match.group(1))
        return None
    
    def evaluate_example(self, prompt: str, correct_count: int) -> Dict:
        """Evaluate a single example."""
        # Add system prompt before the task
        system_prompt = "Count the number of words in the following list that match the given type, and put the numerical answer in parentheses.\n\n"
        full_prompt = system_prompt + prompt
        
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            # Use smaller batch size and clear cache after generation
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=True,
                temperature=0.1,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Move outputs to CPU and clear GPU memory
            outputs = outputs.cpu()
            del inputs
            torch.cuda.empty_cache()
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # The response should complete the open parenthesis, so we look for a number followed by ')'
        match = re.search(r'\((\d+)\)', response)
        if not match:
            # Try to find just the number after the open parenthesis
            response_after_open = response[response.find('('):]
            match = re.search(r'(\d+)', response_after_open)
        
        predicted_count = int(match.group(1)) if match else None
        
        return {
            "prompt": full_prompt,  # Include the system prompt in the saved prompt
            "full_response": response,
            "predicted_count": predicted_count,
            "correct_count": correct_count,
            "is_correct": predicted_count == correct_count if predicted_count is not None else False,
            "error_type": self._get_error_type(predicted_count, correct_count)
        }
    
    def _get_error_type(self, predicted: int, correct: int) -> str:
        """Categorize the type of error made by the model."""
        if predicted is None:
            return "no_answer"
        if predicted == correct:
            return "correct"
        if predicted > correct:
            return "overcounting"
        return "undercounting"
    
    def evaluate_dataset(self, dataset: List[Dict], num_examples: int = None) -> Dict:
        """Evaluate the model on the dataset."""
        if num_examples is not None:
            dataset = dataset[:num_examples]
        
        # Print the first prompt that will be passed to the model
        print("\nFirst prompt that will be passed to the model:")
        system_prompt = "Count the number of words in the following list that match the given type, and put the numerical answer in parentheses.\n\n"
        print(system_prompt + dataset[0]["formatted_prompt"])
        print()
        
        results = []
        for example in tqdm(dataset):
            result = self.evaluate_example(
                example["formatted_prompt"],
                example["count"]
            )
            result["category"] = example["type"]
            result["matching_words"] = example["matching_words"]
            result["list_length"] = len(example["list"])
            results.append(result)
            
        # Calculate metrics
        accuracy = np.mean([r["is_correct"] for r in results])
        
        # Per-category metrics
        category_metrics = {}
        for category in set(r["category"] for r in results):
            cat_results = [r for r in results if r["category"] == category]
            category_metrics[category] = {
                "accuracy": np.mean([r["is_correct"] for r in cat_results]),
                "error_types": {
                    "no_answer": len([r for r in cat_results if r["error_type"] == "no_answer"]),
                    "overcounting": len([r for r in cat_results if r["error_type"] == "overcounting"]),
                    "undercounting": len([r for r in cat_results if r["error_type"] == "undercounting"])
                }
            }
        
        # Error analysis by list length
        length_metrics = {}
        for length in sorted(set(r["list_length"] for r in results)):
            length_results = [r for r in results if r["list_length"] == length]
            length_metrics[length] = {
                "accuracy": np.mean([r["is_correct"] for r in length_results]),
                "count": len(length_results)
            }
        
        return {
            "model_name": self.model_name,
            "overall_accuracy": accuracy,
            "category_metrics": category_metrics,
            "length_metrics": length_metrics,
            "error_distribution": {
                "no_answer": len([r for r in results if r["error_type"] == "no_answer"]),
                "overcounting": len([r for r in results if r["error_type"] == "overcounting"]),
                "undercounting": len([r for r in results if r["error_type"] == "undercounting"]),
                "correct": len([r for r in results if r["error_type"] == "correct"])
            },
            "detailed_results": results
        }

def main():
    parser = argparse.ArgumentParser(description="Evaluate LMs on word counting task")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model name")
    parser.add_argument("--num_examples", type=int, default=100, help="Number of examples to evaluate")
    parser.add_argument("--dataset", type=str, default="word_counting_dataset.json", help="Path to dataset")
    parser.add_argument("--output", type=str, default="evaluation_results.json", help="Output file path")
    args = parser.parse_args()
    
    # Load dataset
    with open(args.dataset, 'r') as f:
        dataset = json.load(f)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(args.model)
    
    # Run evaluation
    results = evaluator.evaluate_dataset(dataset, args.num_examples)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\nResults for {args.model}:")
    print(f"Overall accuracy: {results['overall_accuracy']:.2%}")
    
    print("\nCategory accuracies and error types:")
    for category, metrics in results['category_metrics'].items():
        print(f"{category}:")
        print(f"  Accuracy: {metrics['accuracy']:.2%}")
        print("  Error types:")
        for error_type, count in metrics['error_types'].items():
            print(f"    {error_type}: {count}")
    
    print("\nAccuracy by list length:")
    for length, metrics in results['length_metrics'].items():
        print(f"Length {length}: {metrics['accuracy']:.2%} ({metrics['count']} examples)")
    
    print("\nOverall error distribution:")
    total = sum(results['error_distribution'].values())
    for error_type, count in results['error_distribution'].items():
        print(f"{error_type}: {count} ({count/total:.2%})")

if __name__ == "__main__":
    main() 