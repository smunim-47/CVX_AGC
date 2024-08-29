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
For given `n` and `d`, in order to compare the upper bound of circulant and transition matrix with the worst case fixed decoding for graph, run: 
```sh
python MILP.py n d itr 
```
`Itr` is the number of optimization problems run for the circulant matrix case. 

### Optimal Error vs Upper Bound

For given `n` and `d`, to compare the upper bounds of different schemes with the optimal error (for a certain bad straggler set), run:

```sh
python OptvsFixed.py n d itr
```
Again, `Itr` is the number of optimization problems run for the circulant matrix case. 

We find the bad straggler set for which the optimal error is high in the following way:  First we pick a random data subset $i \in [n]$. We then determine the set that consists of all the workers the subset was assigned to. Then we determine another subset which is assigned to the maximum number of workers from the determined set. We then update the set by taking the union of the set and the set of workers the determined subset is assigned to. If the size of the updated set  is greater than $s$, we take a subset of the updated set of size $s$. Otherwise, we continue this process until the size of the set reaches the number of stragglers $s$.  

The algorithm is the following:
<img src="https://github.com/user-attachments/assets/3666de5d-83f1-4d78-825b-bb88195858ee" alt="Algorithm" width="500"/>





