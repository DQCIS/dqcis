import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=1, batch_first=True)

    def forward(self, history_vectors, query_vector, attention_weight_ratio=0.1):
        # Ensure all tensors are on the same device
        device = query_vector.device
        if history_vectors is not None:
            history_vectors = history_vectors.to(device)
        
        query_vector = query_vector.to(device)
        
        # Apply attention mechanism from history to current query
        if history_vectors is not None:
            attn_output, attn_weights = self.attention(query_vector, history_vectors, history_vectors)
        else:
            attn_output = query_vector
            attn_weights = None
        
        # Adjust the attention output to focus more on the rewrite query
        adjusted_output = attention_weight_ratio * query_vector + (1 - attention_weight_ratio) * attn_output
        
        return adjusted_output, attn_weights