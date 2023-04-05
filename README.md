# DOMINO
## Decomposed Mutual Information Optimization for Generalized Context in Meta-Reinforcement Learning

TensorFlow 2.0 implementation of "DOMINO: Decomposed Mutual Information Optimization for Generalized Context in Meta-Reinforcement Learning" (NeuRIPS 2022).

The whole framework is shown as follow:
![DOMINO Framework](pngs/framework.png)

## Method

This paper addresses the multi-confounded challenge by decomposed mutual information optimization for context learning, which explicitly learns a disentangled context to maximize the mutual information between the context and historical trajectories while minimizing the state transition prediction error. 

- [Project webpage](https://sites.google.com/view/dominorl/)

## Instructions

Install required packages with below commands:

```
conda create -n domino python=3.6
pip install -r requirements.txt
```

Train and evaluate agents:

```
python -m run_scripts.run_domino --dataset [hopper,ant,halfcheetah,cripple_ant,cripple_halfcheetah] --normalize_flag
```

## Some Notes

The code in the repository is an TF2.0 version based on CADM and TMCL, which were written in TensorFlow 1.0. I upgraded the code from TF1.0 to TF2.0 due to the poor operability of the TF1.0 code. I suggest reviewing the source code of CADM and TMCL, as well as my implementation of the TF2.0 version of T-MCL, which can be found at https://github.com/YaoMarkMu/TF2.0-Trajectory-MCL. These resources should aid in your understanding and implementation of the code.