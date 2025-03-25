import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DynamicCache


class LMState(object):
    def __init__(self):
        # history
        self.chat_history = []
        self.past_key_values = DynamicCache()
        self.last_token = None
        
        self.prev_chat_history = []
        self.hyp_past_key_values = DynamicCache()
        self.hyp_last_token = None

        # generation
        self.full_generated_ids = []
        self.generated_ids = []
    
    def update_history(self, role: str, new_content: str):
        if self.chat_history and self.chat_history[-1]["role"] == role:
            msg = self.chat_history.pop()
            if role == "system":
                msg = {"role": msg["role"], "content": new_content}
            else:
                msg = {"role": msg["role"], "content": msg["content"] + new_content}
            self.chat_history.append(msg)
        else: 
            self.chat_history.append({"role": role, "content": new_content})

    def reset(self):
        # history
        self.chat_history = []
        self.past_key_values = DynamicCache()
        self.last_token = None

        self.prev_chat_history = []
        self.hyp_past_key_values = DynamicCache()
        self.hyp_last_token = None

        # generation
        self.full_generated_ids = []
        self.generated_ids = []


def multinomial_top_k_sampling(logits, k, temperature=1.0):
    """
    Perform multinomial sampling with top-k filtering on the output logits.

    Parameters:
    logits (torch.Tensor): The output logits of shape (batch_size, vocab_size).
    k (int): The number of top logits to keep.
    temperature (float): Temperature parameter for sampling. Default is 1.0.
    
    Returns:
    torch.Tensor: Sampled indices of shape (batch_size,).
    """

    # Step 1: Apply top-k filtering
    top_k_logits, top_k_indices = torch.topk(logits, k, dim=-1)
    
    # Step 2: Apply temperature
    top_k_logits = top_k_logits / temperature
    
    # Step 3: Convert to probabilities
    top_k_probs = F.softmax(top_k_logits, dim=-1)
    
    # Step 4: Perform multinomial sampling
    sampled_indices = torch.multinomial(top_k_probs, num_samples=1)
    
    # Step 5: Map the sampled indices back to original indices
    final_indices = top_k_indices.gather(dim=-1, index=sampled_indices)
    
    return final_indices.squeeze(-1)
