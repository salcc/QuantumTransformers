import argparse
import json

import wandb

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Start a wandb sweep')
    argparser.add_argument('dataset', type=str, help='name of dataset to train on',
                           choices=['mnist', 'electron-photon', 'quark-gluon', 'imdb'])  # TODO: add medmnist
    argparser.add_argument('--config', type=str, default='config.json', help='path to JSON config file containing hyperparameters to sweep over')
    argparser.add_argument('--quantum', action='store_true', help='whether to use quantum transformers')
    args = argparser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)
    dataset_type = 'text' if args.dataset in ['imdb'] else 'vision'
    sweep_config = {
        'method': 'bayes',
        'metric': {
            'name': 'val_auc',
            'goal': 'maximize'
        },
        'parameters': {
            'dataset': {
                'value': args.dataset
            },
            'quantum': {
                'value': args.quantum
            },
        } | config['common'] | config[dataset_type]['common'] | config[dataset_type][args.dataset],
        'early_terminate': {
            'type': 'hyperband',
            'min_iter': 10
        }
    }
    
    project_name = 'QuantumTransformers-' + args.dataset + ('-quantum' if args.quantum else '')
    sweep_id = wandb.sweep(sweep_config, project=project_name)
    print()
    print("Project name:", project_name)
    print("Sweep ID:", sweep_id)
