import torch
from torch import nn
import pennylane as qml


class QuantumLinear(nn.Module):  # note that input and output dimension are the same (num_qubits)
    def __init__(self, num_qubits, num_qlayers=1, qdevice="default.qubit"):
        super().__init__()

        # print(f"Using {num_qlayers} quantum layers with {num_qubits} qubits on {qdevice}")

        def _circuit(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=range(num_qubits))
            qml.templates.BasicEntanglerLayers(weights, wires=range(num_qubits))
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(num_qubits)]

        dev = qml.device(qdevice, wires=num_qubits)
        qlayer = qml.QNode(_circuit, dev, interface="torch")
        self.linear = qml.qnn.TorchLayer(qlayer, {"weights": (num_qlayers, num_qubits)})

    def forward(self, inputs):
        return self.linear(inputs)


class BaseMultiheadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, f"Embedding dimension ({embed_dim}) should be divisible by number of heads ({num_heads})"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = None
        self.dropout = nn.Dropout(dropout)
        self.o_proj = None

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        # x.shape = (batch_size, seq_len, embed_dim)
        assert embed_dim == self.embed_dim, f"Input embedding dimension ({embed_dim}) should match layer embedding dimension ({self.embed_dim})"

        # Compute Q, K, V matrices
        qkv = self.qkv_proj(x)
        # qkv.shape = (batch_size, seq_len, embed_dim * 3)
        qkv = qkv.reshape(batch_size, seq_len, self.num_heads, 3 * self.head_dim)
        # qkv.shape = (batch_size, seq_len, num_heads, 3 * head_dim)
        qkv = qkv.permute(0, 2, 1, 3)
        # qkv.shape = (batch_size, num_heads, seq_len, 3 * head_dim)
        q, k, v = qkv.chunk(3, dim=-1)
        # q.shape = k.shape = v.shape = (batch_size, num_heads, seq_len, head_dim)

        # Compute scaled dot-product attention
        attn_logits = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        # attn_logits.shape = (batch_size, num_heads, seq_len, seq_len)
        attn = attn_logits.softmax(dim=-1)
        # attn.shape = (batch_size, num_heads, seq_len, seq_len)
        attn = self.dropout(attn)

        # Compute output
        values = attn @ v
        # values.shape = (batch_size, num_heads, seq_len, head_dim)
        values = values.permute(0, 2, 1, 3)
        # values.shape = (batch_size, seq_len, num_heads, head_dim)
        values = values.reshape(batch_size, seq_len, embed_dim)
        # values.shape = (batch_size, seq_len, embed_dim)
        values = self.o_proj(values)
        # values.shape = (batch_size, seq_len, embed_dim)

        return values


class ClassicalMultiheadSelfAttention(BaseMultiheadSelfAttention):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__(embed_dim, num_heads, dropout)

        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.o_proj = nn.Linear(embed_dim, embed_dim)


class QuantumMultiheadSelfAttention(BaseMultiheadSelfAttention):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__(embed_dim, num_heads, dropout)

        self.qkv_proj = QuantumLinear(embed_dim * 3)
        self.o_proj = QuantumLinear(embed_dim)


class BaseFeedForward(nn.Module):
    def __init__(self, hidden_size, mlp_hidden_size, dropout):
        super().__init__()

        self.fc1 = nn.Linear(hidden_size, mlp_hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(mlp_hidden_size, hidden_size)

    def forward(self, x):
        raise NotImplementedError


class ClassicalFeedForward(BaseFeedForward):
    def __init__(self, hidden_size, mlp_hidden_size, dropout):
        super().__init__(hidden_size, mlp_hidden_size, dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x


class QuantumFeedForward(BaseFeedForward):
    def __init__(self, hidden_size, mlp_hidden_size, dropout):
        super().__init__(hidden_size, mlp_hidden_size, dropout)

        self.vqc = QuantumLinear(mlp_hidden_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.vqc(x)
        x = self.dropout(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x


class BaseTransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_hidden_size, dropout):
        super().__init__()

        self.attn_norm = nn.LayerNorm(hidden_size)
        self.attn = None
        self.attn_dropout = nn.Dropout(dropout)

        self.mlp_norm = nn.LayerNorm(hidden_size)
        self.mlp = None
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


class ClassicalTransformerBlock(BaseTransformerBlock):
    def __init__(self, hidden_size, num_heads, mlp_hidden_size, dropout):
        super().__init__(hidden_size, num_heads, mlp_hidden_size, dropout)

        self.attn = ClassicalMultiheadSelfAttention(hidden_size, num_heads, dropout)
        self.mlp = ClassicalFeedForward(hidden_size, mlp_hidden_size, dropout)


class QuantumTransformerBlock(BaseTransformerBlock):
    def __init__(self, hidden_size, num_heads, mlp_hidden_size, dropout):
        super().__init__(hidden_size, num_heads, mlp_hidden_size, dropout)

        self.attn = QuantumMultiheadSelfAttention(hidden_size, num_heads, dropout)
        self.mlp = QuantumFeedForward(hidden_size, mlp_hidden_size, dropout)


class BaseVisionTransformer(nn.Module):
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

        self.transformer_blocks = None

        self.layer_norm = nn.LayerNorm(hidden_size)

        self.linear = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        if self.channels_last:
            x = x.permute(0, 3, 1, 2)

        # Split image into patches
        x = self.patch_embedding(x)
        x = x.flatten(start_dim=2)
        x = x.transpose(1, 2)

        # CLS token
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)

        # Positional embedding
        x = self.dropout(x + self.pos_embedding)

        # Transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)

        # Layer normalization
        x = self.layer_norm(x)

        # Get the classification token
        x = x[:, 0]

        # Classification logits
        x = self.linear(x)

        return x


class ClassicalVisionTransformer(BaseVisionTransformer):
    def __init__(self, img_size, num_channels, num_classes, patch_size, hidden_size, num_heads, num_transformer_blocks, mlp_hidden_size,
                 dropout=0.1, channels_last=False):
        super().__init__(img_size, num_channels, num_classes, patch_size, hidden_size, num_heads, num_transformer_blocks, mlp_hidden_size,
                         dropout=dropout, channels_last=channels_last)

        self.transformer_blocks = nn.ModuleList([ClassicalTransformerBlock(hidden_size, num_heads, mlp_hidden_size, dropout)
                                                 for _ in range(num_transformer_blocks)])


class QuantumVisionTransformer(BaseVisionTransformer):
    def __init__(self, img_size, num_channels, num_classes, patch_size, hidden_size, num_heads, num_transformer_blocks, mlp_hidden_size,
                 dropout=0.1, channels_last=False):
        super().__init__(img_size, num_channels, num_classes, patch_size, hidden_size, num_heads, num_transformer_blocks, mlp_hidden_size,
                         dropout=dropout, channels_last=channels_last)

        self.transformer_blocks = nn.ModuleList([QuantumTransformerBlock(hidden_size, num_heads, mlp_hidden_size, dropout)
                                                 for _ in range(num_transformer_blocks)])


class BaseTransformer(nn.Module):
    def __init__(self, num_tokens, num_classes, num_transformer_blocks, hidden_size, num_heads, mlp_hidden_size, dropout=0.1):
        super().__init__()

        self.token_embedding = nn.Embedding(num_tokens, hidden_size)
        self.pos_embedding = nn.Embedding(num_tokens, hidden_size)
        self.dropout = nn.Dropout(dropout)

        self.transformer_blocks = None

        self.layer_norm = nn.LayerNorm(hidden_size)

        self.linear = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Token embedding
        x = self.token_embedding(x)

        # Positional embedding
        batch_size, seq_len, embed_dim = x.shape
        positions = torch.arange(seq_len, device=x.device).expand(batch_size, seq_len)
        x = x + self.pos_embedding(positions)

        # Dropout
        x = self.dropout(x)

        # Transformer blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)

        # Layer normalization
        x = self.layer_norm(x)

        # Global average pooling
        x = x.mean(dim=1)

        # Classification logits
        x = self.linear(x)

        return x


class ClassicalTransformer(BaseTransformer):
    def __init__(self, num_tokens, num_classes, num_transformer_blocks, hidden_size, num_heads, mlp_hidden_size, dropout=0.1):
        super().__init__(num_tokens, num_classes, num_transformer_blocks, hidden_size, num_heads, mlp_hidden_size, dropout=0.1)

        self.transformer_blocks = nn.ModuleList([ClassicalTransformerBlock(hidden_size, num_heads, mlp_hidden_size, dropout)
                                                 for _ in range(num_transformer_blocks)])


class QuantumTransformer(BaseTransformer):
    def __init__(self, num_tokens, num_classes, num_transformer_blocks, hidden_size, num_heads, mlp_hidden_size, dropout=0.1):
        super().__init__(num_tokens, num_classes, num_transformer_blocks, hidden_size, num_heads, mlp_hidden_size, dropout=0.1)

        self.transformer_blocks = nn.ModuleList([QuantumTransformerBlock(hidden_size, num_heads, mlp_hidden_size, dropout)
                                                 for _ in range(num_transformer_blocks)])
