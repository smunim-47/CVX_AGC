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
To generate ciruclant and transition matrices for d computation load and n workers, run:
```sh
python Generate_Endcoding_Matrix.py n d itr
```
This script genertes a random d regular graph with n vertices, runs itr circulant matrix optimizations (then takes the best result) and 1 transition matrix optmization. It ouputs the optimal encoding matrices and the corresponding improvement factors. 
