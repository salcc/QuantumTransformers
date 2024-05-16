# Quantum Transformers

This project explores how the [Transformer](https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)) architecture can be executed on quantum computers. In particular, the focus is on the adaptation of the [Vision Transformer](https://en.wikipedia.org/wiki/Vision_transformer) for the analysis of high-energy physics data.

The relevance of the work is accentuated by the upcoming start of operation of the [High Luminosity Large Hadron Collider (HL-LHC)](https://hilumilhc.web.cern.ch/content/hl-lhc-project) at the end of this decade. The program will produce enormous quantities of data, which in turn will require vast computing resources. A promising approach for dealing with this huge amount of data could be the application of quantum machine learning (QML), which could reduce the time complexity of classical algorithms by running on quantum computers and obtain better accuracies.

The work has been undertaken by [Marçal Comajoan Cara](https://salcc.github.io/) as part of [Google Summer of Code](https://summerofcode.withgoogle.com) 2023 with the [ML4SCI organization](https://ml4sci.org/).

You can read more details about the project in [this blog post](https://salcc.github.io/blog/gsoc23), which includes a summary of the results and the potential work that could be done in the future building on it. The research done has also been published in a [paper](https://doi.org/10.3390/axioms13050323) (see below for the citation).

## Structure

The folder structure of the project is as follows:

- `quantum_transformers/`: the library code for the quantum transformers, as well as for loading the data (`datasets.py`) and training the models (`training.py`).
  - `quantum_transformers/qmlperfcomp/`: subproject to compare the performance of different quantum machine learning frameworks. In particular, I evaluated [PennyLane](https://pennylane.ai/) and [TensorCircuit](https://tensorcircuit.readthedocs.io/) (spoiler: TensorCircuit is much faster).
- `notebooks`: the notebooks used for evaluating the models and showing their usage and performance. Each notebook is named after the dataset it uses.
  - `visualizations.ipynb`: notebook visualizing the image datasets.
  - `classical/`: classical counterparts as baselines.
  - `quantum/`: the quantum transformers. Additionally, `qvit_cerrat_et_al.ipynb` is a notebook trying to reproduce the results of ["Quantum Vision Transformers" by Cerrat et al.](https://arxiv.org/abs/2106.03173), although unsuccessfully.
- `hpopt/`: hyperparameter optimization scripts. The folder contains a README with instructions on how to run them.


## Datasets

The architectures have been evaluated on the following datasets:

- [MNIST Digits](http://yann.lecun.com/exdb/mnist/), as a toy dataset for rapid prototyping
- [Quark-Gluon](https://arxiv.org/abs/1902.08276), one of the main high-energy physics datasets used in the project, which contains images of the recordings of the CMS detector of quark and gluon jets.
- [Electron-Photon](https://arxiv.org/abs/1807.11916), the other main high-energy physics dataset used in the project, which contains images of the recordings of the CMS detector of electron and photon showers.
- [IMDb Reviews](https://www.tensorflow.org/datasets/catalog/imdb_reviews), as a toy dataset for evaluating the non-vision transformers for text.

The datasets are downloaded automatically when loading them for the first time. Note that they require a lot of disk space and can take a long time to preprocess.

## Installation

First, install [Python](https://www.python.org/downloads/) if you don't have it already. Then, to install the project together with the dependencies, run the following command in the root folder:

```
pip install -e .
```

## Usage

After installation, you can run the notebooks in the `notebooks` folder. You can also import the library in your own code (`import quantum_transformers`).

## Citation

If you find this project helpful for your research, please cite our [paper](https://doi.org/10.3390/axioms13050323):

```
@Article{axioms13050323,
AUTHOR = {Comajoan Cara, Marçal and Dahale, Gopal Ramesh and Dong, Zhongtian and Forestano, Roy T. and Gleyzer, Sergei and Justice, Daniel and Kong, Kyoungchul and Magorsch, Tom and Matchev, Konstantin T. and Matcheva, Katia and Unlu, Eyup B.},
TITLE = {Quantum Vision Transformers for Quark–Gluon Classification},
JOURNAL = {Axioms},
VOLUME = {13},
YEAR = {2024},
NUMBER = {5},
ARTICLE-NUMBER = {323},
URL = {https://www.mdpi.com/2075-1680/13/5/323},
ISSN = {2075-1680},
DOI = {10.3390/axioms13050323}
}
```


## Acknowledgements

I would like to thank the mentors and fellow contributors from ML4SCI, especially [Sergei Gleyzer](http://sergeigleyzer.com/), for supervising the project and providing guidance and support. I would also like to thank the ML4SCI organization for giving me the opportunity to work on this project, and Google for sponsoring it and supporting the Google Summer of Code program. Likewise, I would also like to thank the [United States National Energy Research Scientific Computing Center (NERSC)](https://www.nersc.gov/) for providing me with the computing resources to run the experiments. Finally, I also want to thank all the developers of the open-source software that I have used for this project.

## License

The project is licensed under the [GNU General Public License v3.0](LICENSE.md).

## Contact

If you have any questions, feel free to email me at [mcomajoancara@gmail.com](mailto:mcomajoancara@gmail.com).
