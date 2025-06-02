import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import os

def load_patching_results(filepath: str) -> Dict:
    """Load activation patching results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def plot_layer_effects(results: Dict, model_name: str, save_dir: str = "plots"):
    """
    Create visualizations of layer-wise effects from activation patching.
    
    Args:
        results: Dictionary containing patching analysis results
        model_name: Name of the model analyzed
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Get layer-wise effects
    direct_effects = results["average_direct_effects"]
    indirect_effects = results["average_indirect_effects"]
    num_layers = len(direct_effects)
    
    # Plot direct and indirect effects
    plt.figure(figsize=(12, 6))
    
    # Create line plots
    plt.plot(range(num_layers), direct_effects, 
             label="Direct Effects", marker='o')
    plt.plot(range(num_layers), indirect_effects, 
             label="Indirect Effects", marker='o')
    
    # Highlight top layers
    for layer in results["top_direct_layers"]:
        plt.axvline(x=layer, color='r', alpha=0.2)
    for layer in results["top_indirect_layers"]:
        plt.axvline(x=layer, color='b', alpha=0.2)
    
    plt.title(f"Layer-wise Effects in {model_name}")
    plt.xlabel("Layer")
    plt.ylabel("Effect Size")
    plt.legend()
    plt.grid(True)
    
    plt.savefig(os.path.join(save_dir, f"{model_name}_layer_effects.png"))
    plt.close()
    
    # Create heatmap of effects
    plt.figure(figsize=(12, 4))
    
    data = np.vstack([direct_effects, indirect_effects])
    sns.heatmap(data, 
                cmap='RdBu_r',
                center=0,
                xticklabels=range(num_layers),
                yticklabels=['Direct', 'Indirect'])
    
    plt.title(f"Effect Heatmap for {model_name}")
    plt.xlabel("Layer")
    
    plt.savefig(os.path.join(save_dir, f"{model_name}_effect_heatmap.png"))
    plt.close()

def plot_example_effects(results: Dict, model_name: str, save_dir: str = "plots"):
    """
    Create visualizations of effects across different examples.
    
    Args:
        results: Dictionary containing patching analysis results
        model_name: Name of the model analyzed
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Get effects for all examples
    all_effects = results["all_example_effects"]
    num_examples = len(all_effects)
    num_layers = len(all_effects[0]["direct_effects"])
    
    # Create matrices of effects
    direct_matrix = np.zeros((num_examples, num_layers))
    indirect_matrix = np.zeros((num_examples, num_layers))
    
    for i, effects in enumerate(all_effects):
        direct_matrix[i] = effects["direct_effects"]
        indirect_matrix[i] = effects["indirect_effects"]
    
    # Plot heatmap of effects across examples
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Direct effects
    sns.heatmap(direct_matrix, 
                cmap='RdBu_r',
                center=0,
                xticklabels=range(num_layers),
                ax=ax1)
    ax1.set_title("Direct Effects Across Examples")
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Example")
    
    # Indirect effects
    sns.heatmap(indirect_matrix,
                cmap='RdBu_r',
                center=0,
                xticklabels=range(num_layers),
                ax=ax2)
    ax2.set_title("Indirect Effects Across Examples")
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Example")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{model_name}_example_effects.png"))
    plt.close()
    
    # Plot distribution of effects per layer
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Direct effects
    sns.boxplot(data=direct_matrix, ax=ax1)
    ax1.set_title("Distribution of Direct Effects by Layer")
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Effect Size")
    
    # Indirect effects
    sns.boxplot(data=indirect_matrix, ax=ax2)
    ax2.set_title("Distribution of Indirect Effects by Layer")
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Effect Size")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{model_name}_effect_distributions.png"))
    plt.close()

def main():
    # Load and visualize results for each model
    models = ["gpt2", "gemma", "phi2"]
    
    for model_name in models:
        results_file = f"{model_name}_patching_results.json"
        
        if not os.path.exists(results_file):
            print(f"No results file found for {model_name}")
            continue
            
        print(f"\nVisualizing results for {model_name}...")
        
        # Load results
        results = load_patching_results(results_file)
        
        # Create visualizations
        plot_layer_effects(results, model_name)
        plot_example_effects(results, model_name)
        
        # Print summary statistics
        print(f"\nTop layers with direct effects:")
        for layer in results["top_direct_layers"]:
            effect = results["average_direct_effects"][layer]
            print(f"Layer {layer}: {effect:.3f}")
            
        print(f"\nTop layers with indirect effects:")
        for layer in results["top_indirect_layers"]:
            effect = results["average_indirect_effects"][layer]
            print(f"Layer {layer}: {effect:.3f}")

if __name__ == "__main__":
    main() 