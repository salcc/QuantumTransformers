# Hyperparameter Optimization

## Start a hyperparameter sweep

To start the hyperparameter sweep, for example, for the `mnist` dataset, run:

```bash
python sweep.py mnist
```

This will output the project name (`QuantumTransformers-mnist` in this case) and the sweep ID, which can be used to start the agents.

Use `--config <config_file>` to use another configuration file path (default: `config.json`).

Use the `--quantum` flag to use quantum transformers.

## Start one agent

> **Note**
> If you want to run multiple agents in parallel, see the section below.

To start an agent to execute the created hyperparameter sweep, for example, for the `mnist` dataset, run (replace `XXXXXXXX` with the sweep ID):

```bash
python agent.py QuantumTransformers-mnist XXXXXXXX
```

The results can be visualized in the [Weights & Biases](https://wandb.ai) dashboard.

## Start multiple agents in parallel (NERSC Perlmutter)

To execute multiple runs in parallel, multiple agents have to be started.
To do so in NERSC Perlmutter using SLURM, use the `nersc_agents.sh` script.
For example, to start 10 agents for the `mnist` dataset, run (replace `XXXXXXXX` with the sweep ID):

```bash
bash nersc_agents.sh QuantumTransformers-mnist XXXXXXXX 10
```

Note that the script has to be modified to use your own account and project, and possibly different options if using a different system.

