import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """ Custom Layer Normalization module.
    This implementation is for educational purposes to show the inner workings
    of layer normalization. i.e., "torch.nn.LayerNorm" should be used.
    
    Args:
        emb_dim (int): The embedding (features) dimension.
    """
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, in_x):
        """
        Args:
            in_x : The input tensor. (batch_size x num_tokens x emb_dim)

        Returns:
             The normalized output tensor. (batch_size x num_tokens x emb_dim)
        """
        mean = in_x.mean(dim=-1, keepdim=True)
        var = in_x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (in_x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

class GELU(nn.Module):
    """ Gaussian Error Linear Unit (GELU) activation function.
        Use torch.nn.GELU instead for simplicity/production code.
    """
    def __init__(self):
        super().__init__()

    def forward(self, in_x):
        return 0.5 * in_x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (in_x + 0.044715 * torch.pow(in_x, 3))
        ))


class FeedForward(nn.Module):
    """ Feed-forward neural network block.
        Two linear layers with a GELU activation in between.

        Args:
            num_layers (dict): A dictionary containing the configuration for the layers,
                            including 'emb_dim'.
        """
    def __init__(self, num_layers):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_layers["emb_dim"], 4 * num_layers["emb_dim"]), ## Expansion
            GELU(), ## Activation
            nn.Linear(4 * num_layers["emb_dim"], num_layers["emb_dim"]), ## Contraction
        )

    def forward(self, x):
        return self.layers(x)


class MultiHeadAttention(nn.Module):
    """ Multi-Head Self-Attention module with a causal mask.

    It computes scaled dot-product attention over multiple heads
    in parallel. A causal mask is applied to prevent positions from
    attending to subsequent positions, which is crucial for auto-regressive
    models like GPT.

    Args:
        d_in (int): input features.
        d_out (int): output features.
        context_length (int): maximum length of the input sequences.
        dropout (float): dropout rate (causal mask).
        num_heads (int): number of attention heads.
        qkv_bias (bool): whether to include bias in the query, key, and value
                         projections. Defaults to False.
    """
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1)
        )

    def forward(self, in_x):
        """
        Args:
            in_x: Input tensor of shape
                    (batch_size, num_tokens, d_in).

        Returns:
            torch.Tensor: Context vector of shape
                          (batch_size, num_tokens, d_out).
        """
        b, num_tokens, d_in = in_x.shape

        keys = self.W_key(in_x) # Shape: (b, num_tokens, d_out)
        queries = self.W_query(in_x)
        values = self.W_value(in_x)

        # Implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim) 
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2) 
        
        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec) # optional projection

        return context_vec
    
class TransformerBlock(nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=num_layers["emb_dim"],
            d_out=num_layers["emb_dim"],
            context_length=num_layers["context_length"],
            num_heads=num_layers["n_heads"],
            dropout=num_layers["drop_rate"],
            qkv_bias=num_layers["qkv_bias"])
        self.ff = FeedForward(num_layers)
        self.norm1 = LayerNorm(num_layers["emb_dim"])
        self.norm2 = LayerNorm(num_layers["emb_dim"])
        self.drop_shortcut = nn.Dropout(num_layers["drop_rate"])

    def forward(self, in_x):
        # Shortcut connection for attention block
        shortcut = in_x
        x = self.norm1(in_x)
        x = self.att(x)  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)    
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x
       

class GPTModel(nn.Module):
    def __init__(self, num_layers):
        super().__init__()
        self.tok_emb = nn.Embedding(num_layers["vocab_size"], num_layers["emb_dim"])
        self.pos_emb = nn.Embedding(num_layers["context_length"], num_layers["emb_dim"])
        self.drop_emb = nn.Dropout(num_layers["drop_rate"])
        
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(num_layers) for _ in range(num_layers["n_layers"])])
        
        self.final_norm = LayerNorm(num_layers["emb_dim"])
        self.out_head = nn.Linear(
            num_layers["emb_dim"], num_layers["vocab_size"], bias=False
        )

    def forward(self, in_x):
        batch_size, seq_len = in_x.shape
        tok_embeds = self.tok_emb(in_x)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_x.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits