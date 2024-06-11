# CVX_AGC
Source code for the work "Approximate Gradient Coding Using Convex Optimization" 


## Requirements

- Python 3.11.5
- CVXPY 1.4.1
- Gurobi Optimizer 11.0.2
- NetworkX 3.3
- NumPy 1.26.0

## Instructions

### Generate Encoding Matrix
To generate cirucluant and transition matrices for `d` computation load and `n` workers, run:
```sh
python Generate_Endcoding_Matrix.py n d itr
```
This script genertes a random d regular graph with n vertices, runs `itr` circulant matrix optimizations (then takes the best result) and 1 transition matrix optmization with the random graph as the underlying graph. It ouputs the optimal encoding matrices (circulant and transition) and the corresponding improvement factors. 

### Worst Case Fixed Decoding Error vs Upper Bound
For given `n` and `d`, in order to compare the upper bound of circulant transition matrix with the worst case fixed decoding for graph, run: 
```sh
python MILP.py n d itr 
```
`Itr` is the number of optimization problems run for the circulant matrix case. 
