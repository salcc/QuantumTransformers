import time

import jax
import jax.numpy as jnp
import flax.linen
import flax.training.train_state
import optax
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


class TrainState(flax.training.train_state.TrainState):
    # See https://flax.readthedocs.io/en/latest/guides/dropout.html.
    key: jax.random.KeyArray


@jax.jit
def train_step(state: TrainState, batch, dropout_key):
    dropout_train_key = jax.random.fold_in(key=dropout_key, data=state.step)

    def loss_fn(params):
        logits = state.apply_fn(
            {'params': params},
            x=batch['input'],
            train=True,
            rngs={'dropout': dropout_train_key}
        )
        if logits.shape[1] <= 2:
            if logits.shape[1] == 2:
                logits = logits[:, 1]
            loss = optax.sigmoid_binary_cross_entropy(logits=logits, labels=batch['label']).mean()
        else:
            loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=batch['label']).mean()
        # return loss, logits
        return loss
    # grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    # (loss, logits), grads = grad_fn(state.params)
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state


@jax.jit
def eval_step(state: TrainState, batch):
    logits = state.apply_fn(
        {'params': state.params},
        x=batch['input'],
        train=False,
        rngs={'dropout': state.key}
    )
    if logits.shape[1] <= 2:
        if logits.shape[1] == 2:
            logits = logits[:, 1]
        loss = optax.sigmoid_binary_cross_entropy(logits=logits, labels=batch['label']).mean()
    else:
        loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=batch['label']).mean()
    results = {'loss': loss, 'logits': logits, 'labels': batch['label']}
    return results


def train_and_evaluate(model: flax.linen.Module, train_dataloader, val_dataloader, num_classes: int,
                       num_epochs: int, learning_rate: float = 1e-3, seed: int = 42, verbose: bool = False) -> None:
    """Trains the given model on the given dataloaders for the given parameters"""
    root_key = jax.random.PRNGKey(seed=seed)
    root_key, params_key, dropout_key = jax.random.split(key=root_key, num=3)

    dummy_batch = next(iter(train_dataloader))[0]
    input_shape = dummy_batch[0].shape
    input_dtype = dummy_batch[0].dtype
    batch_size = len(dummy_batch)
    x = jnp.zeros(shape=(batch_size,) + tuple(input_shape), dtype=input_dtype)  # Dummy input

    variables = model.init(params_key, x, train=False)

    if verbose:
        print(jax.tree_map(lambda x: x.shape, variables))

    optimizer = optax.adam(learning_rate)

    state = TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        key=dropout_key,
        tx=optimizer
    )

    best_val_auc, best_epoch = 0.0, 0
    start_time = time.time()
    for epoch in range(num_epochs):
        with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1:3}/{num_epochs}", unit="batch", bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}') as progress_bar:
            for x, y in train_dataloader:
                step_start_time = time.time()
                batch = {'input': x, 'label': y}
                state = train_step(state, batch, dropout_key)
                progress_bar.update(1)
                if verbose:
                    print(f" Step {state.step+1}/{len(train_dataloader)}: {time.time()-step_start_time:.2f}s")

            logits, labels = [], []
            val_loss = 0.0
            for x, y in val_dataloader:
                batch = {'input': x, 'label': y}
                results = eval_step(state, batch)
                logits.append(results['logits'])
                labels.append(results['labels'])
                val_loss += results['loss']
            val_loss /= len(val_dataloader)
            logits = jnp.concatenate(logits)
            y_true = jnp.concatenate(labels)
            if num_classes == 2:
                y_pred = [jax.nn.sigmoid(l) for l in logits]
            else:
                y_pred = [jax.nn.softmax(l) for l in logits]
            val_auc = 100.0 * roc_auc_score(y_true, y_pred, multi_class='ovr')

            progress_bar.set_postfix_str(f"Loss = {val_loss:.4f}, AUC = {val_auc:.2f}%")

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_epoch = epoch + 1

    print(f"TOTAL TIME = {time.time()-start_time:.2f}s")
    print(f"BEST AUC = {best_val_auc:.2f}% AT EPOCH {best_epoch}")
