# pytorch_net

This repository provides classes for 
- Efficient construction of neural networks with PyTorch, including multilayer perceptron (MLP), LSTM, CNN, WideResNet, model ensembles, etc.
- Easy manipulation of networks, including addition and removal of layers and neurons, training of networks, simplification of networks, loading and saving models as a dictionary. 
See [tutorial.ipynb](https://github.com/tailintalent/pytorch_net/blob/master/Tutorial.ipynb) for a simple demonstration.


## Requirements:
- [PyTorch](https://pytorch.org/) >= 0.4.1
- scikit-learn
- sympy >= 1.3 for symbolic layers


## Some projects using this library:
- [LAMP](https://github.com/snap-stanford/lamp): ICLR 2023 spotlight
- [LE-PDE](https://github.com/snap-stanford/le_pde): NeurIPS 2022
- [AI Physicist](https://github.com/tailintalent/AI_physicist): Physical Review E
- [GIB](https://github.com/snap-stanford/GIB): NeurIPS 2020
- [Causal learning](https://github.com/tailintalent/causal): ICML 2019 Time Series Workshop, best poster award
- [Meta-learning autoencoder](https://github.com/tailintalent/mela)
