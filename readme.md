# Simulation-based Bayesian Inference from Privacy Protected Data

This repository contains code for our paper titled "Simulation-based Bayesian Inference from Privacy Protected Data". Our work addresses the challenge of performing valid Bayesian inference from datasets that have been privacy-protected using differential privacy mechanisms.

## Repository Contents
- `main_baseline.py`: Core script demonstrating baseline experiments.
- `SNPE_lib.py`: Neural conditional density estimation using various flow architectures.
- `Dataset.py`: Data generation and privacy perturbation mechanisms for synthetic and real datasets.
- `sir_c.cpp`: Efficient simulator for the stochastic SIR model implemented in C++.
- `smc_abc.py`: Implement for SMC-ABC algorithm.


### Compilation

To use the efficient SIR model simulator, compile the provided C++ file using your preferred compiler:

```bash
g++ -O3 -shared -std=c++11 -fPIC sir_c.cpp -o libsir_c.so
```

Adjust the compilation command for your system and compiler accordingly.

## Usage

### Running baseline experiments

Execute the baseline model experiments with:

```bash
python main_baseline.py --gpu 1 --eps 1.0 --data 1
```

Parameters:
- `--gpu`: GPU device (set to `0` for CPU).
- `--eps`: Privacy budget (epsilon) for differential privacy.
- `--data`: Dataset type (`0`: SIR model, `1`: Linear Regression, `2`: Log-linear model).

