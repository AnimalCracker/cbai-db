import torch
from transformer_lens import HookedTransformer
from typing import List, Dict, Tuple, Optional
import numpy as np
from tqdm import tqdm
import json
import argparse
import re
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from einops import rearrange, reduce, repeat
import os

@dataclass
class CountExample:
    """Class to hold a counting example and its analysis results."""
    prompt: str
    category: str
    word_list: List[str]
    matching_words: List[str]
    true_count: int
    predicted_count: int
    token_positions: List[int]  # Positions of list words in tokenized input
    answer_position: int  # Position of the answer token

class RunningCountAnalyzer:
    def __init__(self, model_name: str = "gpt2-small", device: str = "cuda"):
        """Initialize model and setup for running count analysis."""
        self.model_name = model_name
        self.device = device
        
        print(f"Loading {model_name} with TransformerLens...")
        self.model = HookedTransformer.from_pretrained(
            model_name,
            device=device,
        )
        
        self.num_layers = self.model.cfg.n_layers
        self.d_model = self.model.cfg.d_model
        
    def parse_example(self, prompt: str) -> CountExample:
        """Parse a counting example into its components."""
        # Extract category
        category_match = re.search(r"Type: (\w+)", prompt)
        if not category_match:
            raise ValueError("Could not find category in prompt")
        category = category_match.group(1)
        
        # Extract word list
        list_match = re.search(r"List: \[(.*?)\]", prompt)
        if not list_match:
            raise ValueError("Could not find word list in prompt")
        word_list = [w.strip("' ") for w in list_match.group(1).split(",")]
        
        # Extract answer - much simpler!
        answer_match = re.search(r"Answer: \((.+?)\)", prompt)
        if not answer_match:
            # Handle incomplete prompts that end with "Answer: ("
            if prompt.strip().endswith("Answer: ("):
                true_count = 0  # Use 0 as placeholder for incomplete prompts
            else:
                raise ValueError("Could not find answer in prompt")
        else:
            answer_text = answer_match.group(1)
            # Convert to int, handling "None" case
            if answer_text.isdigit():
                true_count = int(answer_text)
            else:
                true_count = -1  # Use -1 for "None" or invalid answers
        
        # Get tokens
        tokens = self.model.to_tokens(prompt)[0]
        
        # Find word positions - simplified approach
        # Just look for positions after each comma or opening bracket in the list
        list_start = prompt.find("List: [")
        word_positions_in_text = []
        current_pos = list_start + 7  # Start after "List: ["
        
        for word in word_list:
            word_start = prompt.find(f"'{word}'", current_pos)
            if word_start != -1:
                word_positions_in_text.append(word_start + 1)  # Position after quote
                current_pos = word_start + len(word) + 2  # Move past this word
        
        # Convert text positions to token positions
        token_positions = []
        for text_pos in word_positions_in_text:
            # Find the token that contains this text position
            token_pos = len(self.model.to_tokens(prompt[:text_pos])[0]) - 1
            token_positions.append(token_pos)
        
        # Find answer position - use last position for models like Gemma where sequences get truncated
        tokens = self.model.to_tokens(prompt)[0]
        answer_position = len(tokens) - 1
        
        # Run model to get predicted count
        with torch.no_grad():
            logits = self.model(prompt)
            
        # Get the model's prediction at the answer position
        predicted_count = self.get_number_prediction(logits, answer_position)
        
        return CountExample(
            prompt=prompt,
            category=category,
            word_list=word_list,
            matching_words=[],  # Will be filled in by caller
            true_count=true_count,
            predicted_count=predicted_count,
            token_positions=token_positions,
            answer_position=answer_position
        )
    
    def run_with_hooks(self, 
                      prompt: str, 
                      layer_idx: int = None,
                      position_idx: int = None,
                      patch_activations: torch.Tensor = None) -> Tuple[torch.Tensor, Dict]:
        """
        Run model with activation hooks for patching or recording.
        
        Args:
            prompt: Input prompt
            layer_idx: Layer to patch (if patching)
            position_idx: Position to patch (if patching)
            patch_activations: Activations to patch in (if patching)
        """
        activation_cache = {}
        
        def hook_fn(act, hook):
            # Record activations
            activation_cache[hook.name] = act.detach()
            
            # Patch if requested
            if (layer_idx is not None and 
                position_idx is not None and 
                patch_activations is not None and
                f"blocks.{layer_idx}" in hook.name and
                hook.name.endswith(('mlp.hook_post', 'attn.hook_post'))):
                
                # Patch the activation at the specified position
                act[:, position_idx:position_idx+1, :] = patch_activations
            
            return act
        
        # Run model with hooks
        with torch.no_grad():
            logits = self.model.run_with_hooks(
                prompt,
                fwd_hooks=[(lambda name: True, hook_fn)]
            )
        
        return logits, activation_cache
    
    def analyze_pair(self, 
                    example_a: CountExample, 
                    example_b: CountExample) -> Dict:
        """
        Analyze a pair of examples by patching activations.
        
        Args:
            example_a: First example (source of activations)
            example_b: Second example (target for patching)
        """
        # Use the minimum number of positions to avoid index errors
        max_positions = min(len(example_a.token_positions), len(example_b.token_positions))
        
        results = {
            "mlp_effects": np.zeros((self.num_layers, max_positions)),
            "attn_effects": np.zeros((self.num_layers, max_positions)),
            "source_count": example_a.true_count,
            "target_count": example_b.true_count,
            "num_positions": max_positions
        }
        
        # Get base activations for both examples
        _, cache_a = self.run_with_hooks(example_a.prompt)
        base_logits, cache_b = self.run_with_hooks(example_b.prompt)
        
        # For each layer and position
        for layer in range(self.num_layers):
            for pos_idx in range(max_positions):
                token_pos = example_b.token_positions[pos_idx]
                
                # Patch MLP activations
                mlp_key = f"blocks.{layer}.mlp.hook_post"
                if mlp_key in cache_a:
                    # Extract the activation at the specific position
                    source_activation = cache_a[mlp_key][:, token_pos:token_pos+1, :].clone()
                    
                    patched_logits, _ = self.run_with_hooks(
                        example_b.prompt,
                        layer_idx=layer,
                        position_idx=token_pos,
                        patch_activations=source_activation
                    )
                    
                    # Calculate effect on prediction
                    base_pred = self.get_number_prediction(base_logits, example_b.answer_position)
                    patched_pred = self.get_number_prediction(patched_logits, example_b.answer_position)
                    
                    results["mlp_effects"][layer, pos_idx] = abs(patched_pred - base_pred)
                
                # Patch attention activations
                attn_key = f"blocks.{layer}.attn.hook_post"
                if attn_key in cache_a:
                    # Extract the activation at the specific position
                    source_activation = cache_a[attn_key][:, token_pos:token_pos+1, :].clone()
                    
                    patched_logits, _ = self.run_with_hooks(
                        example_b.prompt,
                        layer_idx=layer,
                        position_idx=token_pos,
                        patch_activations=source_activation
                    )
                    
                    patched_pred = self.get_number_prediction(patched_logits, example_b.answer_position)
                    results["attn_effects"][layer, pos_idx] = abs(patched_pred - base_pred)
        
        return results
    
    def get_number_prediction(self, logits: torch.Tensor, position: int) -> int:
        """Get the predicted number from logits at a position."""
        number_tokens = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        number_token_ids = []
        for num in number_tokens:
            token_id = self.model.to_tokens(num)[0][-1]  # Get the last token
            number_token_ids.append(token_id)
        
        number_token_ids = torch.tensor(number_token_ids).to(self.device)
        number_logits = logits[0, position][number_token_ids]
        predicted_idx = number_logits.argmax()
        return int(number_tokens[predicted_idx])
    
    def plot_effects(self, results: Dict, save_path: str):
        """Plot the effects of patching at different layers and positions."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))
        
        # Plot MLP effects
        sns.heatmap(
            results["mlp_effects"],
            ax=ax1,
            cmap='viridis',
            xticklabels=[f"Pos {i}" for i in range(results["mlp_effects"].shape[1])],
            yticklabels=[f"Layer {i}" for i in range(results["mlp_effects"].shape[0])]
        )
        ax1.set_title(f"MLP Patching Effects (Source Count: {results['source_count']}, Target Count: {results['target_count']})")
        
        # Plot attention effects
        sns.heatmap(
            results["attn_effects"],
            ax=ax2,
            cmap='viridis',
            xticklabels=[f"Pos {i}" for i in range(results["attn_effects"].shape[1])],
            yticklabels=[f"Layer {i}" for i in range(results["attn_effects"].shape[0])]
        )
        ax2.set_title("Attention Patching Effects")
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Analyze running count representations")
    parser.add_argument("--model", type=str, default="gpt2", help="Model name")
    parser.add_argument("--example_pairs", type=str, required=True, help="Path to example pairs JSON")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save results")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                      help="Device to run analysis on")
    args = parser.parse_args()
    
    # Load example pairs
    print(f"Loading example pairs from {args.example_pairs}...")
    with open(args.example_pairs, 'r') as f:
        example_pairs = json.load(f)
    
    # Initialize analyzer
    analyzer = RunningCountAnalyzer(args.model, args.device)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Analyze pairs
    all_results = []
    for i, (correct_ex, incorrect_ex) in enumerate(tqdm(example_pairs[:10])):  # Start with 10 pairs
        try:
            # Parse examples
            example_a = analyzer.parse_example(correct_ex)
            example_b = analyzer.parse_example(incorrect_ex)
            
            # Run analysis
            results = analyzer.analyze_pair(example_a, example_b)
            
            # Plot results
            analyzer.plot_effects(
                results,
                os.path.join(args.output_dir, f"effects_pair_{i}.png")
            )
            
            all_results.append(results)
            
        except Exception as e:
            print(f"\nError processing pair {i}: {str(e)}")
            continue
    
    # Save aggregate results
    if all_results:
        # Find the maximum number of positions across all results
        max_positions = max(r["num_positions"] for r in all_results)
        
        # Pad all results to the same size
        padded_mlp_effects = []
        padded_attn_effects = []
        
        for r in all_results:
            mlp_effect = r["mlp_effects"]
            attn_effect = r["attn_effects"]
            
            # Pad with zeros if needed
            if mlp_effect.shape[1] < max_positions:
                mlp_padded = np.zeros((analyzer.num_layers, max_positions))
                mlp_padded[:, :mlp_effect.shape[1]] = mlp_effect
                mlp_effect = mlp_padded
                
            if attn_effect.shape[1] < max_positions:
                attn_padded = np.zeros((analyzer.num_layers, max_positions))
                attn_padded[:, :attn_effect.shape[1]] = attn_effect
                attn_effect = attn_padded
                
            padded_mlp_effects.append(mlp_effect)
            padded_attn_effects.append(attn_effect)
        
        # Now we can safely average
        avg_mlp_effects = np.mean(padded_mlp_effects, axis=0)
        avg_attn_effects = np.mean(padded_attn_effects, axis=0)
        
        aggregate_results = {
            "average_mlp_effects": avg_mlp_effects.tolist(),
            "average_attn_effects": avg_attn_effects.tolist(),
            "num_pairs_analyzed": len(all_results),
            "max_positions": max_positions
        }
        
        with open(os.path.join(args.output_dir, "aggregate_results.json"), 'w') as f:
            json.dump(aggregate_results, f, indent=2)
        
        # Plot aggregate results
        plt.figure(figsize=(12, 16))
        
        plt.subplot(2, 1, 1)
        sns.heatmap(avg_mlp_effects, cmap='viridis')
        plt.title("Average MLP Effects Across All Pairs")
        
        plt.subplot(2, 1, 2)
        sns.heatmap(avg_attn_effects, cmap='viridis')
        plt.title("Average Attention Effects Across All Pairs")
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "aggregate_effects.png"))
        plt.close()

if __name__ == "__main__":
    main() 