# Code for 'Variational Bayesian Optimal Experimental Design'

### Installation
First, install Python 3 and `pytorch>=1.0`. For full instructions on installing Pytorch on your system,
 see https://pytorch.org/.
Second, install this fork of pyro. This can most easily be accomplished by first changing into the directory 
containing this file, and then running

    > pip install -e .[dev]
    
On some systems, the `--user` option may be necessary.

### General guidance
The experiments for this paper live in `examples/contrib/oed`. Experiments are run by launching a
job that produces a pickled output in `run_outputs`, followed by running a results parser to examine
the output of the run.

### EIG estimation accuracy
The following reproduces the EIG estimation experiments, with new random seeds. The case tags used in the paper
are: `ab_test`, `preference`, `mixed_effects_regression`, `extrapolation`.

    > python3 examples/contrib/oed/eig_estimation_benchmarking.py --case-tags=<case> --name=<your filename without extension>
    > python3 examples/contrib/oed/results_parser_eig_estimation_benchmarking.py --fnames=<same filename>
    
**Note**: some issues with `mixed_effects_regression` at the moment.

### Convergence rates
Use the following commands to reproduce plots from the paper.

    > python3 examples/contrib/oed/convergence_n.py --seed=972219904 --fname=<your filename wihtout extension>
    > python3 examples/contrib/oed/results_parser_convergence_n.py --fnames=<same filename>
    
    > python3 examples/contrib/oed/convergence_k.py --seed=960926848 --fname=<your filename wihtout extension>
    > python3 examples/contrib/oed/results_parser_convergence_k.py --fnames=<same filename>
    
    > python3 examples/contrib/oed/convergence_proportion.py --seed=921163584 --fname=<your filename wihtout extension>
    > python3 examples/contrib/oed/results_parser_convergence_proportion.py --fnames=<same filename>
    
    > python3 examples/contrib/oed/convergence_tradeoff.py --seed=1002875328 --fname=<your filename wihtout extension>
    > python3 examples/contrib/oed/results_parser_convergence_tradeoff.py --fnames=<same filename>
    
    > python3 examples/contrib/oed/convergence_vnmc.py --seed=782823872 --fname=<your filename wihtout extension>
    > python3 examples/contrib/oed/results_parser_convergence_vnmc.py --fnames=<same filename>
    
    
### End-to-end sequential experiments
To reproduce the CES experiment use the following commands

    > python3 examples/contrib/oed/ces.py --num-steps=20 --seed=734252288 --typs=marginal --lengthscale=20 --name=<marginal file name>
    > python3 examples/contrib/oed/ces.py --num-steps=20 --seed=466574528 --typs=rand --name=<rand file name>
    > python3 examples/contrib/oed/ces.py --num-steps=20 --seed=668716672 --typs=nmc --lengthscale=20 --name=<nmc file name>
    > python3 examples/contrib/oed/results_parser_ces.py --fnames=<marginal,rand,nmc>
