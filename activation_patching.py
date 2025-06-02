import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Tuple, Optional
import numpy as np
from tqdm import tqdm
import json
import argparse
import re

class ActivationPatcher:
    def __init__(self, model_name: str, device: str = "cuda"):
        """Initialize model and tokenizer for activation patching analysis."""
        self.model_name = model_name
        self.device = device
        
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            output_hidden_states=True  # Enable hidden states output
        )
        
        if device != "cuda":
            self.model = self.model.to(device)
        
        # Get model architecture info
        self.num_layers = self.model.config.num_hidden_layers
        self.hidden_size = self.model.config.hidden_size
        
    def get_hidden_states(self, prompt: str) -> Tuple[torch.Tensor, List[torch.Tensor], int]:
        """Run forward pass and return logits, hidden states, and target token index."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Find the position of the answer number
        answer_match = re.search(r"Answer: \((\d+)\)", prompt)
        if not answer_match:
            raise ValueError("Could not find answer number in prompt")
            
        # Get the token position of the number
        answer_start = prompt.index(f"Answer: ({answer_match.group(1)})")
        prefix_tokens = self.tokenizer(prompt[:answer_start], return_tensors="pt").input_ids
        target_idx = prefix_tokens.size(1)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            
        # Get logits and all hidden states
        logits = outputs.logits
        hidden_states = outputs.hidden_states  # Tuple of tensors (num_layers + 1, batch_size, seq_len, hidden_size)
        
        return logits, hidden_states, target_idx
    
    def patch_hidden_states(self, 
                          source_states: List[torch.Tensor],
                          target_states: List[torch.Tensor],
                          layer_idx: int,
                          token_idx: Optional[int] = None) -> List[torch.Tensor]:
        """
        Create patched hidden states by copying states from source to target at specified layer.
        If token_idx is provided, only patch that specific token position.
        """
        patched_states = list(target_states)  # Make a copy
        
        if token_idx is not None:
            # Patch specific token
            source_state = source_states[layer_idx][:, token_idx:token_idx+1, :]
            target_shape = patched_states[layer_idx][:, token_idx:token_idx+1, :].shape
            if source_state.shape == target_shape:
                patched_states[layer_idx][:, token_idx:token_idx+1, :] = source_state
            else:
                print(f"Warning: Shape mismatch at layer {layer_idx}, token {token_idx}")
                print(f"Source shape: {source_state.shape}, Target shape: {target_shape}")
        else:
            # Patch entire layer
            source_state = source_states[layer_idx]
            target_shape = patched_states[layer_idx].shape
            if source_state.shape == target_shape:
                patched_states[layer_idx] = source_state
            else:
                print(f"Warning: Shape mismatch at layer {layer_idx}")
                print(f"Source shape: {source_state.shape}, Target shape: {target_shape}")
            
        return patched_states
    
    def run_causal_trace(self,
                        correct_example: str,
                        incorrect_example: str) -> Dict[str, np.ndarray]:
        """
        Perform causal tracing analysis between correct and incorrect examples.
        
        Args:
            correct_example: Prompt where model counts correctly
            incorrect_example: Prompt where model counts incorrectly
            
        Returns:
            Dictionary containing effect sizes for each layer
        """
        # Get hidden states for both examples
        correct_logits, correct_states, correct_target_idx = self.get_hidden_states(correct_example)
        incorrect_logits, incorrect_states, incorrect_target_idx = self.get_hidden_states(incorrect_example)
        
        # Original logits at target positions
        orig_correct_logits = correct_logits[:, correct_target_idx, :]
        orig_incorrect_logits = incorrect_logits[:, incorrect_target_idx, :]
        
        # Store effects for each layer
        direct_effects = np.zeros(self.num_layers)
        indirect_effects = np.zeros(self.num_layers)
        
        # Analyze each layer
        for layer_idx in range(self.num_layers):
            # Patch correct -> incorrect
            patched_states = self.patch_hidden_states(
                correct_states, incorrect_states, layer_idx, incorrect_target_idx)
            
            # Run forward pass with patched states
            with torch.no_grad():
                patched_logits = self.model.forward(
                    inputs_embeds=patched_states[0],
                    hidden_states=patched_states[1:],
                    output_hidden_states=True
                ).logits
            
            # Calculate direct effect (change in correct prediction probability)
            direct_effect = (
                patched_logits[:, incorrect_target_idx, :] - orig_incorrect_logits
            ).abs().mean().item()
            
            # Calculate indirect effect
            indirect_effect = (
                orig_correct_logits - patched_logits[:, correct_target_idx, :]
            ).abs().mean().item()
            
            direct_effects[layer_idx] = direct_effect
            indirect_effects[layer_idx] = indirect_effect
            
        return {
            "direct_effects": direct_effects,
            "indirect_effects": indirect_effects
        }
    
    def analyze_running_count(self, example_pairs: List[Tuple[str, str]]) -> Dict:
        """
        Analyze multiple example pairs to find evidence of running count representations.
        
        Args:
            example_pairs: List of (correct_example, incorrect_example) tuples
            
        Returns:
            Analysis results including layer-wise effects and potential count-tracking layers
        """
        all_effects = []
        
        for correct_ex, incorrect_ex in tqdm(example_pairs):
            try:
                effects = self.run_causal_trace(correct_ex, incorrect_ex)
                all_effects.append(effects)
            except Exception as e:
                print(f"\nError processing example pair: {str(e)}")
                continue
        
        if not all_effects:
            raise ValueError("No valid results from any example pairs")
        
        # Aggregate effects across examples
        avg_direct = np.mean([e["direct_effects"] for e in all_effects], axis=0)
        avg_indirect = np.mean([e["indirect_effects"] for e in all_effects], axis=0)
        
        # Find layers with strongest effects
        direct_peaks = np.argsort(avg_direct)[-3:][::-1]  # Top 3 layers
        indirect_peaks = np.argsort(avg_indirect)[-3:][::-1]
        
        return {
            "average_direct_effects": avg_direct.tolist(),  # Convert to list for JSON serialization
            "average_indirect_effects": avg_indirect.tolist(),
            "top_direct_layers": direct_peaks.tolist(),
            "top_indirect_layers": indirect_peaks.tolist(),
            "all_example_effects": [
                {
                    "direct_effects": e["direct_effects"].tolist(),
                    "indirect_effects": e["indirect_effects"].tolist()
                }
                for e in all_effects
            ]
        }

def main():
    parser = argparse.ArgumentParser(description="Run activation patching analysis")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., gpt2)")
    parser.add_argument("--example_pairs", type=str, required=True, help="Path to example pairs JSON file")
    parser.add_argument("--output", type=str, required=True, help="Path to save results")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                      help="Device to run analysis on (cuda/cpu)")
    args = parser.parse_args()
    
    # Load example pairs
    print(f"Loading example pairs from {args.example_pairs}...")
    with open(args.example_pairs, 'r') as f:
        example_pairs = json.load(f)
    
    print(f"Running analysis on {len(example_pairs)} example pairs...")
    
    # Initialize patcher
    patcher = ActivationPatcher(args.model, device=args.device)
    
    # Run analysis
    results = patcher.analyze_running_count(example_pairs)
    
    # Save results
    print(f"\nSaving results to {args.output}...")
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\nAnalysis complete! Summary of findings:")
    print("\nTop layers with direct effects:")
    for layer in results["top_direct_layers"]:
        print(f"Layer {layer}: {results['average_direct_effects'][layer]:.3f}")
    
    print("\nTop layers with indirect effects:")
    for layer in results["top_indirect_layers"]:
        print(f"Layer {layer}: {results['average_indirect_effects'][layer]:.3f}")

if __name__ == "__main__":
    main() 