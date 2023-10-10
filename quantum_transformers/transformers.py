from typing import Literal, Callable, Optional
import flax.linen as nn
import jax.numpy as jnp

from quantum_transformers.quantum_layer import QuantumLayer

# See:
# - https://nlp.seas.harvard.edu/annotated-transformer/
# - https://github.com/rdisipio/qtransformer/blob/main/qtransformer.py
# - https://github.com/google-research/vision_transformer/blob/main/vit_jax/models_vit.py


class MultiHeadSelfAttention(nn.Module):
    hidden_size: int
    num_heads: int
    dropout: float = 0.0

    quantum_w_shape: tuple = (1,)
    quantum_circuit: Optional[Callable] = None

    @nn.compact
    def __call__(self, x, deterministic):
        batch_size, seq_len, hidden_size = x.shape
        # x.shape = (batch_size, seq_len, hidden_size)
        assert hidden_size == self.hidden_size, f"Input hidden size ({hidden_size}) does not match layer hidden size ({self.hidden_size})"
        assert hidden_size % self.num_heads == 0, f"Hidden size ({hidden_size}) must be divisible by the number of heads ({self.num_heads})"
        head_dim = hidden_size // self.num_heads

        if self.quantum_circuit is None:
            q, k, v = [
                proj(x).reshape(batch_size, seq_len, self.num_heads, head_dim).swapaxes(1, 2)
                for proj, x in zip([nn.Dense(features=hidden_size),
                                    nn.Dense(features=hidden_size),
                                    nn.Dense(features=hidden_size)], [x, x, x])
            ]
        else:
            q, k, v = [
                proj(x).reshape(batch_size, seq_len, self.num_heads, head_dim).swapaxes(1, 2)
                for proj, x in zip([QuantumLayer(num_qubits=hidden_size, w_shape=self.quantum_w_shape, circuit=self.quantum_circuit),
                                    QuantumLayer(num_qubits=hidden_size, w_shape=self.quantum_w_shape, circuit=self.quantum_circuit),
                                    QuantumLayer(num_qubits=hidden_size, w_shape=self.quantum_w_shape, circuit=self.quantum_circuit)], [x, x, x])
            ]

        # Compute scaled dot-product attention
        attn_logits = (q @ k.swapaxes(-2, -1)) / jnp.sqrt(head_dim)
        # attn_logits.shape = (batch_size, num_heads, seq_len, seq_len)
        attn = nn.softmax(attn_logits, axis=-1)
        # attn.shape = (batch_size, num_heads, seq_len, seq_len)
        attn = nn.Dropout(rate=self.dropout)(attn, deterministic=deterministic)

        # Compute output
        values = attn @ v
        # values.shape = (batch_size, num_heads, seq_len, head_dim)
        values = values.swapaxes(1, 2).reshape(batch_size, seq_len, hidden_size)
        # values.shape = (batch_size, seq_len, hidden_size)
        if self.quantum_circuit is None:
            x = nn.Dense(features=hidden_size)(values)
        else:
            x = QuantumLayer(num_qubits=hidden_size, w_shape=self.quantum_w_shape, circuit=self.quantum_circuit)(values)
        # x.shape = (batch_size, seq_len, hidden_size)

        return x


class FeedForward(nn.Module):
    hidden_size: int
    mlp_hidden_size: int
    dropout: float = 0.0

    quantum_w_shape: tuple = (1,)
    quantum_circuit: Optional[Callable] = None

    @nn.compact
    def __call__(self, x, deterministic):
        x = nn.Dense(features=self.mlp_hidden_size)(x)
        if self.quantum_circuit is not None:
            x = QuantumLayer(num_qubits=self.mlp_hidden_size, w_shape=self.quantum_w_shape, circuit=self.quantum_circuit)(x)
        x = nn.Dropout(rate=self.dropout)(x, deterministic=deterministic)
        x = nn.gelu(x)
        x = nn.Dense(features=self.hidden_size)(x)
        return x


class TransformerBlock(nn.Module):
    hidden_size: int
    num_heads: int
    mlp_hidden_size: int
    dropout: float = 0.0

    quantum_w_shape: tuple = (1,)
    quantum_attn_circuit: Optional[Callable] = None
    quantum_mlp_circuit: Optional[Callable] = None

    @nn.compact
    def __call__(self, x, deterministic):
        attn_output = nn.LayerNorm()(x)
        attn_output = MultiHeadSelfAttention(hidden_size=self.hidden_size, num_heads=self.num_heads, dropout=self.dropout,
                                             quantum_circuit=self.quantum_attn_circuit)(attn_output, deterministic=deterministic)
        attn_output = nn.Dropout(rate=self.dropout)(attn_output, deterministic=deterministic)
        x = x + attn_output

        y = nn.LayerNorm()(x)
        y = FeedForward(hidden_size=self.hidden_size, mlp_hidden_size=self.mlp_hidden_size,
                        quantum_circuit=self.quantum_mlp_circuit)(y, deterministic=deterministic)
        y = nn.Dropout(rate=self.dropout)(y, deterministic=deterministic)

        return x + y


class Transformer(nn.Module):
    num_tokens: int
    max_seq_len: int
    num_classes: int
    hidden_size: int
    num_heads: int
    num_transformer_blocks: int
    mlp_hidden_size: int
    dropout: float = 0.0

    quantum_w_shape: tuple = (1,)
    quantum_attn_circuit: Optional[Callable] = None
    quantum_mlp_circuit: Optional[Callable] = None

    @nn.compact
    def __call__(self, x, train):
        # Token embedding
        x = nn.Embed(num_embeddings=self.num_tokens, features=self.hidden_size)(x)
        # x.shape = (batch_size, seq_len, hidden_size)

        # Positional embedding
        x += nn.Embed(num_embeddings=self.max_seq_len, features=self.hidden_size)(jnp.arange(x.shape[1]))

        # Dropout
        x = nn.Dropout(rate=self.dropout)(x, deterministic=not train)

        # Transformer blocks
        for _ in range(self.num_transformer_blocks):
            x = TransformerBlock(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                mlp_hidden_size=self.mlp_hidden_size,
                dropout=self.dropout,
                quantum_attn_circuit=self.quantum_attn_circuit,
                quantum_mlp_circuit=self.quantum_mlp_circuit
            )(x, deterministic=not train)

        # Layer normalization
        x = nn.LayerNorm()(x)

        # Global average pooling
        x = jnp.mean(x, axis=1)
        # x.shape = (batch_size, hidden_size)

        # Classification logits
        x = nn.Dense(self.num_classes)(x)
        # x.shape = (batch_size, num_classes)

        return x


def posemb_sincos_2d(sqrt_num_steps, hidden_size, temperature=10_000., dtype=jnp.float32):
    """2D sin-cos position embedding. Follows the MoCo v3 logic."""
    # Code adapted from https://github.com/google-research/big_vision/blob/184d1201eb34abe7da84fc69f84fd89a06ad43c4/big_vision/models/vit.py#L33.
    y, x = jnp.mgrid[:sqrt_num_steps, :sqrt_num_steps]

    assert hidden_size % 4 == 0, f"Hidden size ({hidden_size}) must be divisible by 4 for 2D sin-cos position embedding"
    omega = jnp.arange(hidden_size // 4) / (hidden_size // 4 - 1)
    omega = 1. / (temperature**omega)
    y = jnp.einsum("m,d->md", y.flatten(), omega)
    x = jnp.einsum("m,d->md", x.flatten(), omega)
    pe = jnp.concatenate([jnp.sin(x), jnp.cos(x), jnp.sin(y), jnp.cos(y)], axis=1)
    return jnp.asarray(pe, dtype)[None, :, :]


class VisionTransformer(nn.Module):
    num_classes: int
    patch_size: int
    hidden_size: int
    num_heads: int
    num_transformer_blocks: int
    mlp_hidden_size: int
    dropout: float = 0.1
    pos_embedding: Literal['none', 'learn', 'sincos'] = 'learn'
    classifier: Literal['token', 'gap'] = 'gap'
    channels_last: bool = True

    quantum_w_shape: tuple = (1,)
    quantum_attn_circuit: Optional[Callable] = None
    quantum_mlp_circuit: Optional[Callable] = None

    @nn.compact
    def __call__(self, x, train):
        assert x.ndim == 4, f"Input must be 4D, got {x.ndim}D ({x.shape})"

        if not self.channels_last:
            x = x.transpose((0, 3, 1, 2))
        # x.shape = (batch_size, height, width, num_channels)
        # Note that JAX's Conv expects the input to be in the format (batch_size, height, width, num_channels)

        batch_size, height, width, _ = x.shape
        assert height == width, f"Input must be square, got {height}x{width}"
        img_size = height
        num_steps = (img_size // self.patch_size) ** 2

        # Splitting an image into patches and linearly projecting these flattened patches can be
        # simplified as a single convolution operation, where both the kernel size and the stride size
        # are set to the patch size.
        x = nn.Conv(
            features=self.hidden_size,
            kernel_size=(self.patch_size, self.patch_size),
            strides=self.patch_size,
            padding="VALID"
        )(x)
        # x.shape = (batch_size, sqrt(num_steps), sqrt(num_steps), hidden_size)
        sqrt_num_steps = x.shape[1]
        x = jnp.reshape(x, (batch_size, num_steps, self.hidden_size))
        # x.shape = (batch_size, num_steps, hidden_size)

        # Positional embedding
        if self.pos_embedding == 'learn':
            x += self.param('pos_embedding', nn.initializers.normal(stddev=1 / jnp.sqrt(self.hidden_size)),
                            (1, num_steps, self.hidden_size), x.dtype)
        elif self.pos_embedding == 'sincos':
            x += posemb_sincos_2d(sqrt_num_steps, self.hidden_size, dtype=x.dtype)
        elif self.pos_embedding == 'none':
            pass
        else:
            raise ValueError(f"Unknown positional embedding type: {self.pos_embedding}")
        # x.shape = (batch_size, num_steps, hidden_size)

        if self.classifier == 'token':
            # CLS token
            cls_token = self.param('cls', nn.initializers.zeros, (1, 1, self.hidden_size))
            cls_token = jnp.tile(cls_token, (batch_size, 1, 1))
            x = jnp.concatenate([cls_token, x], axis=1)
            num_steps += 1
            # x.shape = (batch_size, num_steps, hidden_size)

        # Dropout
        x = nn.Dropout(rate=self.dropout)(x, deterministic=not train)
        # x.shape = (batch_size, num_steps, hidden_size)

        # Transformer blocks
        for _ in range(self.num_transformer_blocks):
            x = TransformerBlock(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                mlp_hidden_size=self.mlp_hidden_size,
                dropout=self.dropout,
                quantum_attn_circuit=self.quantum_attn_circuit,
                quantum_mlp_circuit=self.quantum_mlp_circuit
            )(x, deterministic=not train)

        # Layer normalization
        x = nn.LayerNorm()(x)
        # x.shape = (batch_size, num_steps, hidden_size)

        if self.classifier == 'token':
            # Get the classifcation token
            x = x[:, 0]
        elif self.classifier == 'gap':
            # Global average pooling
            x = jnp.mean(x, axis=1)
        else:
            raise ValueError(f"Unknown classifier type: {self.classifier}")
        # x.shape = (batch_size, hidden_size)

        # Classification logits
        x = nn.Dense(features=self.num_classes)(x)
        # x.shape = (batch_size, num_classes)

        return x
