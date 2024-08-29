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
Again, `Itr` is the number of optimization problems run for the circulant matrix case. The algorithm to find the bad straggler set is the following: 

$\begin{algorithm}
\caption{Finding the straggler set for subsection V-B}\label{alg:example}
\begin{algorithmic}[1]
\STATE \textbf{Input:} Number of stragglers $s = a\delta$, where $a \in \mathbb{N}$
\STATE \textbf{Output:} A set of stragglers $[n] \setminus \mathcal{F}$ of size $s$ for which the optimal error is relatively high 
\STATE Pick a random $i \in [n]$ and set $\mathcal{F} =  [n]\setminus\text{supp}(\mathbf{B}(i,:))$ and $\mathcal{P} = \{i\}$

\WHILE{$|[n]  \setminus \mathcal{F}| < s$}

\STATE Find $h$ such that $|([n]\setminus \mathcal{F}) \cap \text{supp}(\mathbf{B}(h,:))| \ge |([n]\setminus \mathcal{F}) \cap \text{supp}(\mathbf{B}(j,:))|$ for each $j \in \mathcal{P}$
\IF{$| ([n]\setminus \mathcal{F}) \cup \text{supp}(\mathbf{B}(h,:))| > s$}
\STATE Choose $\mathcal{H} \subset \text{supp}(\mathbf{B}(h,:)) \setminus ([n] \setminus \mathcal{F})$ such that $|([n] \setminus \mathcal{F}) \cup \mathcal{H}| = s$

\STATE Set $\mathcal{F} = [n] \setminus (([n] \setminus \mathcal{F}) \cup \mathcal{H})$

\ELSE
\STATE Set $\mathcal{F} = [n] \setminus (([n] \setminus \mathcal{F}) \cup \text{supp}(\mathbf{B}(h,:)))$
\STATE set $\mathcal{P} = \mathcal{P} \cup \{h\}$
\ENDIF

\ENDWHILE



 
\RETURN $[n] \setminus \mathcal{F}$

\end{algorithmic}
\end{algorithm}$
