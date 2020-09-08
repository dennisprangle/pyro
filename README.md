# Pharmacokinetic model using Foster et al approach

The repository is code used in the paper "Bayesian experimental design without posterior calculations: an adversarial approach" (https://arxiv.org/abs/1904.05703).
In particular it performs experimental design for a pharmacokinetic example
by implementing the approach of the paper "A Unified Stochastic Gradient Approach to Designing Bayesian-Optimal Experiments" (https://arxiv.org/abs/1911.00294).

This code is a fork of the code for the latter paper which in turn is a fork of the Pyro probabilistic programming language.

## Technical notes (from Foster code README)
 - All experiment were run with `torch==1.4.0`
 - Install Pyro from this branch using `pip3 install -e .`

## Pharmacokinetic

To reproduce our pharmacokinetic model analysis, run:

```bash
python3 pk.py --num-steps 963500 --time-budget 100000 --seed 123 --num-parallel 100 --name pk-pce-1hour --estimator pce --device cpu
python3 pk.py --num-steps 1904000 --time-budget 100000 --seed 123 --num-parallel 100 --name pk-pce-2hours --estimator pce --device cpu
```

(n.b. We ran these commands but with time budgets 3600 and 7200,
giving 1 and 2 hours of runtime, and much larger value for num-steps.
The lines above reproduce the number of iterations performed in our analyses,
which will take a different runtime on a different computing environment.)