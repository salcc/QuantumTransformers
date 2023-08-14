import flax.linen

class MLP(flax.linen.Module):
    hidden_size: int

    @flax.linen.compact
    def __call__(self, x, **_):
        x = flax.linen.Dense(self.hidden_size)(x)
        x = flax.linen.relu(x)
        x = flax.linen.Dense(self.hidden_size)(x)
        x = flax.linen.relu(x)
        x = flax.linen.Dense(1)(x)
        return x
