# Hyperparameter Optimization

The code in this directory is intended to be used in NERSC Perlmutter.

Execute submit-ray-cluster.sh to run the hyperparameter optimization for a given dataset and number of trials.
For example, to run 50 trials of hyperparameter optimization for the MNIST dataset, execute the following command:

```bash
bash submit-ray-cluster.sh mnist 50
```

After the two required parameters (dataset name and number of trials), the script accepts the following optional parameters:
- `--quantum`: whether to use quantum transformers

Note that the script has to be modified to use your own project, and possibly different options if using a different system.
