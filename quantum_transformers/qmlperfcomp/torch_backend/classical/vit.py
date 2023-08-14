import torch
from torch import nn

# See:
# - https://nlp.seas.harvard.edu/annotated-transformer/
# - https://github.com/rdisipio/qtransformer/blob/main/qtransformer.py
# - https://github.com/google-research/vision_transformer/blob/main/vit_jax/models_vit.py


class MultiheadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, f"Embedding dimension ({embed_dim}) should be divisible by number of heads ({num_heads})"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        # x.shape = (batch_size, seq_len, embed_dim)
        assert embed_dim == self.embed_dim, f"Input embedding dimension ({embed_dim}) should match layer embedding dimension ({self.embed_dim})"

        q, k, v = [
            proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            for proj, x in zip([self.q_proj, self.k_proj, self.v_proj], [x, x, x])
        ]

        # Compute scaled dot-product attention
        attn_logits = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        # attn_logits.shape = (batch_size, num_heads, seq_len, seq_len)
        attn = attn_logits.softmax(dim=-1)
        # attn.shape = (batch_size, num_heads, seq_len, seq_len)
        attn = self.dropout(attn)

        # Compute output
        values = attn @ v
        # values.shape = (batch_size, num_heads, seq_len, head_dim)
        values = values.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
        # values.shape = (batch_size, seq_len, embed_dim)
        x = self.o_proj(values)
        # x.shape = (batch_size, seq_len, embed_dim)

        return x


class FeedForward(nn.Module):
    def __init__(self, hidden_size, mlp_hidden_size, dropout=0.0):
        super().__init__()

        self.fc1 = nn.Linear(hidden_size, mlp_hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(mlp_hidden_size, hidden_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_hidden_size, dropout=0.0):
        super().__init__()

        self.attn_norm = nn.LayerNorm(hidden_size)
        self.attn = MultiheadSelfAttention(hidden_size, num_heads, dropout)
        self.attn_dropout = nn.Dropout(dropout)

        self.mlp_norm = nn.LayerNorm(hidden_size)
        self.mlp = FeedForward(hidden_size, mlp_hidden_size, dropout)
        self.mlp_dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output = self.attn_norm(x)
        attn_output = self.attn(attn_output)
        attn_output = self.attn_dropout(attn_output)
        x = x + attn_output

        y = self.mlp_norm(x)
        y = self.mlp(y)
        y = self.mlp_dropout(y)

        return x + y


class VisionTransformer(nn.Module):
    def __init__(self, img_size, num_channels, num_classes, patch_size, hidden_size, num_heads, num_transformer_blocks, mlp_hidden_size,
                 dropout=0.1, channels_last=False):
        super().__init__()

        self.channels_last = channels_last

        # Splitting an image into patches and linearly projecting these flattened patches can be
        # simplified as a single convolution operation, where both the kernel size and the stride size
        # are set to the patch size.
        self.patch_embedding = nn.Conv2d(
            in_channels=num_channels,
            out_channels=hidden_size,
            kernel_size=patch_size,
            stride=patch_size
        )
        num_patches = (img_size // patch_size)**2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        num_steps = 1 + num_patches

        self.pos_embedding = nn.Parameter(torch.randn(1, num_steps, hidden_size) * 0.02)
        self.dropout = nn.Dropout(dropout)

        self.transformer_blocks = nn.ModuleList([TransformerBlock(hidden_size, num_heads, mlp_hidden_size, dropout)
                                                 for _ in range(num_transformer_blocks)])

        self.layer_norm = nn.LayerNorm(hidden_size)

        self.linear = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        if self.channels_last:
            x = x.permute(0, 3, 1, 2)
        # x.shape = (batch_size, num_channels, img_size, img_size)
        # Note that PyTorch's Conv2D expects the input to be in the format (batch_size, num_channels, height, width)

        # Split image into patches
        x = self.patch_embedding(x)
        # x.shape = (batch_size, hidden_size, sqrt(num_patches), sqrt(num_patches))
        x = x.flatten(start_dim=2)
        # x.shape = (batch_size, hidden_size, num_patches)
        x = x.transpose(1, 2)
        # x.shape = (batch_size, num_patches, hidden_size)

        # CLS token
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        # x.shape = (batch_size, num_steps, hidden_size)

        # Positional embedding
        x = self.dropout(x + self.pos_embedding)
        # x.shape = (batch_size, num_steps, hidden_size)

        # Transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)

        # Layer normalization
        x = self.layer_norm(x)
        # x.shape = (batch_size, num_steps, hidden_size)

        # Get the classification token
        x = x[:, 0]
        # x.shape = (batch_size, hidden_size)

        # Classification logits
        x = self.linear(x)
        # x.shape = (batch_size, num_classes)

        return x
