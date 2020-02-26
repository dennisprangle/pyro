# Reproduction of experiments for 'A Unified Stochastic Gradient Approach to Designing Bayesian-Optimal Experiments'

This document details the steps to reproduce the experimental results found in the following paper
```
@inproceedings{foster2020unified,
  title={A Unified Stochastic Gradient Approach to Designing Bayesian-Optimal Experiments},
  author={Foster, Adam and Jankowiak, Martin and O'Meara, Matthew and Teh, Yee Whye and Rainforth, Tom},
  booktitle={The 23rd International Conference on Artificial Intelligence and Statistics},
  year={2020},
}
```

### Technical notes
 - All experiment were run with `torch==1.4.0`
 - Install Pyro from this branch using `pip3 install -e .`
 - Design evaluation is known to fail on machines with insufficient CPU memory
 - Seeds were set at random. Omitting the `--seed` argument will create a run with a new random seed
 - Where a number of steps is specified, this number was selected to give a fixed computation time across all methods. By specifying `--num-steps`, one may also run experiments with an equal number of steps. For gradient methods, a step is a gradient step whereas for BO a step involves updating the GP and performing an acquisition. 
 
 ## Death process
 Use the following commands to reproduce Figure 2 in the paper
 ```bash
> python3 death_process_rb.py --num-steps 9600 --name repr-death-ace --num-parallel 100 --seed 591971072 --estimator ace
> python3 death_process_rb.py --num-steps 19000 --name repr-death-pce --num-parallel 100 --seed 774423808 --estimator pce
> python3 death_process_rb.py --num-steps 16600 --name repr-death-ba --num-parallel 100 --seed 1019347136 --estimator posterior
> python3 death_process_bo.py --num-steps 300 --name repr-death-bo --num-parallel 100 --seed 1391488
> python3 results_parser_combine_infograd.py --names repr-death-ace,repr-death-ce,repr-death-ba,repr-death-bo
```
The number of steps were chosen to give 120 seconds compute time.

## Regression
Use the following to reproduce the values presented in Table 1. Note: this experiment can be run on GPU and the default device is `cuda:0`. To run on CPU, specify `--device cpu`.
```bash
> python3 regression.py --num-steps 55771 --seed 229362624 --name repr-regr-ace --estimator ace
> python3 regression.py --num-steps 132746 --seed 773975680 --name repr-regr-pce --estimator pce
> python3 regression.py --num-steps 108785 --seed 155927360 --name repr-regr-ba --estimator posterior
> python3 regression_random_search.py --num-steps 72 --seed 581791424 --name repr-regr-rs
```
For BO, to avoid issues with LBFGS, the 10 runs were performed in series with 72 steps and the following seeds: `1036743680`, `47580224`, `19820544`, `1029449344`, `863347328`,`735120192`,`292830400`,`414648128`,`666109120`, `134552256` using the following command
```bash
> python3 regression_bo.py --num-steps 72 --seed <seed> --name repr-regr-bo-run<i>
```
The BO results can be combined by manipulating the `xi_history` object within the output files.

To evaluate each method (compute the upper and lower bounds on the EIG) use
```bash
> python3 regression_evaluation.py --name <name>
```
This modifies the output file with name `<name>` in place, adding the computed upper and lower bounds. The values presented in the table can then be computed using
```bash
> python3 results_parser_infograd.py --name <name> --summary True
```

## Advertising


## Biomolecular docking



## Constant elasticity of substitution