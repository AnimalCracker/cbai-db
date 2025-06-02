import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import pandas as pd

def load_results(filepath: str) -> Dict:
    with open(filepath, 'r') as f:
        return json.load(f)

def create_accuracy_comparison(results_dict: Dict[str, Dict]) -> None:
    """Create bar plot comparing overall accuracies."""
    models = list(results_dict.keys())
    accuracies = [results["overall_accuracy"] * 100 for results in results_dict.values()]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, accuracies)
    plt.title("Overall Accuracy by Model")
    plt.ylabel("Accuracy (%)")
    plt.xticks(rotation=45)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('accuracy_comparison.png')
    plt.close()

def create_category_heatmap(results_dict: Dict[str, Dict]) -> None:
    """Create heatmap of category accuracies across models."""
    # Extract category accuracies for each model
    category_data = {}
    for model, results in results_dict.items():
        category_accuracies = {cat: metrics["accuracy"] * 100 
                             for cat, metrics in results["category_metrics"].items()}
        category_data[model] = category_accuracies
    
    # Convert to DataFrame
    df = pd.DataFrame(category_data)
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(df, annot=True, fmt='.1f', cmap='YlOrRd')
    plt.title("Category Accuracy by Model (%)")
    plt.tight_layout()
    plt.savefig('category_heatmap.png')
    plt.close()

def create_error_distribution(results_dict: Dict[str, Dict]) -> None:
    """Create stacked bar chart of error distributions."""
    error_types = ["correct", "overcounting", "undercounting", "no_answer"]
    model_errors = []
    
    for model, results in results_dict.items():
        total = sum(results["error_distribution"].values())
        errors = [results["error_distribution"][err_type] / total * 100 for err_type in error_types]
        model_errors.append(errors)
    
    model_errors = np.array(model_errors).T
    models = list(results_dict.keys())
    
    plt.figure(figsize=(10, 6))
    bottom = np.zeros(len(models))
    
    for i, error_type in enumerate(error_types):
        plt.bar(models, model_errors[i], bottom=bottom, label=error_type)
        bottom += model_errors[i]
    
    plt.title("Error Distribution by Model")
    plt.ylabel("Percentage")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('error_distribution.png')
    plt.close()

def create_length_analysis(results_dict: Dict[str, Dict]) -> None:
    """Create line plot of accuracy by list length."""
    plt.figure(figsize=(12, 6))
    
    for model, results in results_dict.items():
        lengths = [int(k) for k in results["length_metrics"].keys()]
        accuracies = [v["accuracy"] * 100 for v in results["length_metrics"].values()]
        plt.plot(lengths, accuracies, marker='o', label=model)
    
    plt.title("Accuracy by List Length")
    plt.xlabel("List Length")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('length_analysis.png')
    plt.close()

def print_summary_statistics(results_dict: Dict[str, Dict]) -> None:
    """Print summary statistics for each model."""
    print("\nSummary Statistics:")
    print("=" * 80)
    
    for model, results in results_dict.items():
        print(f"\nModel: {model}")
        print("-" * 40)
        print(f"Overall Accuracy: {results['overall_accuracy']*100:.1f}%")
        
        # Best and worst categories
        categories = [(cat, metrics["accuracy"]) 
                     for cat, metrics in results["category_metrics"].items()]
        best_cat = max(categories, key=lambda x: x[1])
        worst_cat = min(categories, key=lambda x: x[1])
        
        print(f"Best Category: {best_cat[0]} ({best_cat[1]*100:.1f}%)")
        print(f"Worst Category: {worst_cat[0]} ({worst_cat[1]*100:.1f}%)")
        
        # Error distribution
        total_errors = sum(results["error_distribution"].values())
        print("\nError Distribution:")
        for error_type, count in results["error_distribution"].items():
            print(f"  {error_type}: {count/total_errors*100:.1f}%")
        
        # List length performance
        best_length = max(results["length_metrics"].items(), 
                         key=lambda x: x[1]["accuracy"])
        print(f"\nBest List Length: {best_length[0]} words "
              f"({best_length[1]['accuracy']*100:.1f}% accuracy)")

def main():
    # Load results
    results = {
        "GPT-2": load_results("gpt2_2k_results.json"),
        "Gemma-2B": load_results("gemma_2k_results.json"),
        "Phi-2": load_results("phi2_2k_results.json")
    }
    
    # Create visualizations
    create_accuracy_comparison(results)
    create_category_heatmap(results)
    create_error_distribution(results)
    create_length_analysis(results)
    print_summary_statistics(results)

if __name__ == "__main__":
    main()