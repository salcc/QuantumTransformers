"""Hyperparameter optimization with wandb."""

import argparse

import wandb


def agent() -> None:
    # Perform imports here to avoid warning messages when running only --help
    import tensorflow as tf
    tf.config.set_visible_devices([], device_type='GPU')  # Ensure TF does not see GPU and grab all GPU memory

    from quantum_transformers import datasets
    from quantum_transformers.quantum_layer import get_circuit
    from quantum_transformers.transformers import Transformer, VisionTransformer
    from quantum_transformers.training import train_and_evaluate

    wandb.init()
    c = wandb.config  # Shorter alias for wandb.config
    tf.random.set_seed(c.seed)  # For reproducible data loading

    num_classes = {'imdb': 2, 'mnist': 10, 'electron-photon': 2, 'quark-gluon': 2}  # TODO: add medmnist
    model: Transformer | VisionTransformer
    if c.dataset in ['imdb']:  # Text datasets
        if c.dataset == 'imdb':
            (train_dataloader, val_dataloader, test_dataloader), vocab, _ = datasets.get_imdb_dataloaders(data_dir=c.data_dir, batch_size=c.batch_size,
                                                                                                          max_seq_len=c.max_seq_len, max_vocab_size=c.vocab_size)
        else:
            raise ValueError(f"Unknown dataset {c.dataset}")

        model = Transformer(num_tokens=len(vocab), max_seq_len=c.max_seq_len, num_classes=num_classes[c.dataset],
                            hidden_size=c.hidden_size, num_heads=c.num_heads, num_transformer_blocks=c.num_transformer_blocks, mlp_hidden_size=c.mlp_hidden_size,
                            dropout=c.dropout,
                            quantum_attn_circuit=get_circuit() if c.quantum else None, quantum_mlp_circuit=get_circuit() if c.quantum else None)
    else:  # Vision datasets
        if c.dataset == 'mnist':
            train_dataloader, val_dataloader, test_dataloader = datasets.get_mnist_dataloaders(data_dir=c.data_dir, batch_size=c.batch_size)
        elif c.dataset == 'electron-photon':
            train_dataloader, val_dataloader, test_dataloader = datasets.get_electron_photon_dataloaders(data_dir=c.data_dir, batch_size=c.batch_size)
        elif c.dataset == 'quark-gluon':
            train_dataloader, val_dataloader, test_dataloader = datasets.get_quark_gluon_dataloaders(data_dir=c.data_dir, batch_size=c.batch_size)
        elif c.dataset.startswith('medmnist-'):
            raise NotImplementedError("MedMNIST is not yet supported")
            train_dataloader, val_dataloader, test_dataloader = datasets.get_medmnist_dataloaders(dataset=c.dataset.split('-')[1], data_dir=c.data_dir, batch_size=c.batch_size)
        else:
            raise ValueError(f"Unknown dataset {c.dataset}")

        model = VisionTransformer(num_classes=num_classes[c.dataset], patch_size=c.patch_size,
                                  hidden_size=c.hidden_size, num_heads=c.num_heads, num_transformer_blocks=c.num_transformer_blocks, mlp_hidden_size=c.mlp_hidden_size,
                                  pos_embedding=c.pos_embedding, dropout=c.dropout,
                                  quantum_attn_circuit=get_circuit() if c.quantum else None, quantum_mlp_circuit=get_circuit() if c.quantum else None)

    train_and_evaluate(model=model, train_dataloader=train_dataloader, val_dataloader=val_dataloader, test_dataloader=test_dataloader, num_classes=num_classes[c.dataset],
                       num_epochs=c.num_epochs, lrs_peak_value=c.lrs_peak_value, lrs_warmup_steps=c.lrs_warmup_steps, lrs_decay_steps=c.lrs_decay_steps,
                       seed=c.seed, use_wandb=True)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Start a wandb sweep agent')
    argparser.add_argument('project_name', type=str, help='name of wandb project to run sweep in')
    argparser.add_argument('sweep_id', type=str, help='ID of wandb sweep to run')
    args = argparser.parse_args()

    wandb.agent(args.sweep_id, project=args.project_name, function=agent, count=1)
