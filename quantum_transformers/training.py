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
def train_step(state: TrainState, inputs, labels, key):
    """
    Performs a single training step on the given batch of inputs and labels.

    Args:
        state: The current training state.
        inputs: The batch of inputs.
        labels: The batch of labels.
        key: The random key to use.

    Returns:
        The updated training state.
    """
    key, dropout_key = jax.random.split(key=key)
    dropout_train_key = jax.random.fold_in(key=dropout_key, data=state.step)

    def loss_fn(params):
        logits = state.apply_fn(
            {'params': params},
            x=inputs,
            train=True,
            rngs={'dropout': dropout_train_key}
        )
        if logits.shape[1] <= 2:
            if logits.shape[1] == 2:
                logits = logits[:, 1]
            loss = optax.sigmoid_binary_cross_entropy(logits=logits, labels=labels).mean()
        else:
            loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=labels).mean()
        # return loss, logits
        return loss
    # grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    # (loss, logits), grads = grad_fn(state.params)
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state


@jax.jit
def eval_step(state: TrainState, inputs, labels):
    """
    Performs a single evaluation step on the given batch of inputs and labels.

    Args:
        state: The current training state.
        inputs: The batch of inputs.
        labels: The batch of labels.

    Returns:
        loss: The loss on the given batch.
        logits: The logits on the given batch.
    """
    logits = state.apply_fn(
        {'params': state.params},
        x=inputs,
        train=False,
        rngs={'dropout': state.key}
    )
    if logits.shape[1] <= 2:
        if logits.shape[1] == 2:
            logits = logits[:, 1]
        loss = optax.sigmoid_binary_cross_entropy(logits=logits, labels=labels).mean()
    else:
        loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=labels).mean()
    return loss, logits


def train_and_evaluate(model: flax.linen.Module, train_dataloader, val_dataloader, num_classes: int,
                       num_epochs: int, seed: int = 42, verbose: bool = False) -> None:
    """
    Trains the given model on the given dataloaders for the given hyperparameters.

    The progress and evaluation results are printed to stdout.

    Args:
        model: The model to train.
        train_dataloader: The dataloader for the training set.
        val_dataloader: The dataloader for the validation set.
        num_classes: The number of classes.
        num_epochs: The number of epochs to train for.
        learning_rate: The learning rate to use.
        seed: The seed to use for reproducibility.
        verbose: Whether to print extra information.

    Returns:
        None
    """
    root_key = jax.random.PRNGKey(seed=seed)
    root_key, params_key, train_key = jax.random.split(key=root_key, num=3)

    dummy_batch = next(iter(train_dataloader))[0]
    input_shape = dummy_batch[0].shape
    input_dtype = dummy_batch[0].dtype
    batch_size = len(dummy_batch)
    inputs_batch = jnp.zeros(shape=(batch_size,) + tuple(input_shape), dtype=input_dtype)  # Dummy input

    variables = model.init(params_key, inputs_batch, train=False)

    if verbose:
        print(jax.tree_map(lambda x: x.shape, variables))

    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=1e-3,
        warmup_steps=5_000,
        decay_steps=50_000,
        end_value=0.0
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule),
    )

    state = TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        key=train_key,
        tx=optimizer
    )

    best_val_auc, best_epoch = 0.0, 0
    start_time = time.time()
    for epoch in range(num_epochs):
        with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1:3}/{num_epochs}", unit="batch", bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}') as progress_bar:
            for inputs_batch, labels_batch in train_dataloader:
                step_start_time = time.time()
                state = train_step(state, inputs_batch, labels_batch, train_key)
                progress_bar.update(1)
                if verbose:
                    print(f" Step {state.step+1}/{len(train_dataloader)}: {time.time()-step_start_time:.2f}s")

            logits, labels = [], []
            val_loss = 0.0
            for inputs_batch, labels_batch in val_dataloader:
                loss_batch, logits_batch = eval_step(state, inputs_batch, labels_batch)
                logits.append(logits_batch)
                labels.append(labels_batch)
                val_loss += loss_batch
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
