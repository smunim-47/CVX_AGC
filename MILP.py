#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gurobipy as gp
from gurobipy import GRB
import numpy as np
import networkx as nx
import cvxpy as cp
import matplotlib.pyplot as plt
import os
import sys


time = int(sys.argv[4])

n = int(sys.argv[1])

d = int(sys.argv[2])
    
                

random_seed = 42

#Define a graph

#G = nx.random_regular_expander_graph(n, d)
#G = nx.margulis_gabber_galil_graph(n)
G = nx.random_regular_graph(d, n, seed = random_seed)

#Adjacency matrix of the graph

A_g = nx.to_numpy_array(G)
A_g = (1/d)*A_g


B = A_g








#s = 15  # Example value for s

p = 15
s_list= np.linspace(0, (4*n)/10, p + 1, dtype=int)[1:]
frac_stragglers = s_list/n
ones = np.reshape(np.ones(n), (n,1))
A_g_x_error = np.zeros((len(s_list)))
A_g_opt_error = np.zeros((len(s_list)))
dec_vec = np.zeros((n, len(s_list)))
for o in range(len(s_list)):
            #s = s_list[o]
            print(o)
            # Create a new Gurobi model
            model = gp.Model("quadratic_binary_optimization")
            model.setParam(GRB.Param.TimeLimit, time)

            # Create binary variables x_i
            x = model.addVars(range(n), vtype=GRB.BINARY, name="x")

            # Objective function: x^T B^T B x
            # Compute B^T B
            BTB = B.T @ B

            # Set up the objective function in Gurobi
            objective = gp.QuadExpr()
            for i in range(n):
                for j in range(n):
                    objective += BTB[i, j] * x[i] * x[j]
            objective = objective* -1 
            # Set the objective
            model.setObjective(objective, GRB.MINIMIZE)

            # Add the first constraint: x[i] * x[i] - x[i] = 0 for all i
            # In Gurobi, x[i] is already binary, so this constraint is redundant
            # but if you need to explicitly write it:
            #for i in range(n):
                #model.addConstr(x[i] * x[i] - x[i] == 0, name=f"binary_constraint_{i}")

            # Add the second constraint: sum(x) = n - s
            model.addConstr(gp.quicksum(x[i] for i in range(n)) == n - s_list[o], name="sum_constraint")

            # Optimize the model
            model.optimize()

            # Retrieve and print the results
            if model.status == GRB.OPTIMAL:
                x_opt = model.getAttr('x', x)
                print("Optimal solution:")
                for i in range(n):
                    print(f"x[{i}] = {x_opt[i]}")
            elif model.status == GRB.TIME_LIMIT:
                print("Time limit reached. Best heuristic solution found:")
                if model.SolCount > 0:
                    x_opt = model.getAttr('x', x)
                    for i in range(n):
                        print(f"x[{i}] = {x_opt[i]}")
                else:
                    print("No feasible solution found within the time limit.")
            else:
                print("No optimal solution found")
            print(np.sum(np.array([x[i].X for i in range(n)])))
            r = (1 + (s_list[o]/(n-s_list[o])))*np.array([x[i].X for i in range(n)])
            print(r.shape)
            print(np.sum(r))
            dec_vec[:, o] = r
            r= np.reshape(r, (n,1))
            A_g_F = A_g[:, np.nonzero(r)[0]]
            A_g_opt_error[o] = np.sqrt(n-ones.T@A_g_F@ np.transpose(np.linalg.inv(np.transpose(A_g_F)@A_g_F))@A_g_F.T@ones)

            print(r)
            print(np.linalg.norm(A_g@r - ones , ord = 2))
            A_g_x_error[o] = np.linalg.norm(A_g@r - ones , ord = 2)


#np.save('dec_vecs_'+ str(n)+ '_' + str(d) + '.npy', dec_vec)
#np.save('max error MILP' + str(n)+ '_' + str(d) + '.npy', A_g_x_error)
E_Values, E_VECTORS = np.linalg.eig(A_g)
#v_2 = E_VECTORS[:, 0]
#print(E_VECTORS[:, 0])

E_Values_abs = np.abs(E_Values)
indices = np.argsort(E_Values_abs)[::-1]
n_ev = n-1



evals = np.sort(np.abs(np.linalg.eigvals(A_g)))[::-1]
#print(evals)
IF_g = evals[1]/evals[0]

A_g_upper_bound = (IF_g)* np.sqrt((n*s_list)/(n-s_list))
A_g_v_error = np.zeros((len(s_list), n_ev))

for i in range(len(s_list)):
    #r_v_2 = ((s_list[i]) /( n -s_list[i])) * np.ones((n,1))
    #v_2 = E_VECTORS[:, 1]
    #r_v_2[np.argsort(v_2)[:s_list[i]]] = -1
    #A_g_v2_error[i] = np.linalg.norm(A_g@(r_v_2 + ones) - ones , ord = 2)
    for k in range(n_ev):
        r_v = ((s_list[i]) /( n -s_list[i])) * np.ones((n,1))

        #r_v_c = ((s_list[i]) /( n -s_list[i])) * np.ones((n,1))
        #r_v_t = ((s_list[i]) /( n -s_list[i])) * np.ones((n,1))
        v = E_VECTORS[:, indices[k+1]]

        r_v[np.argsort(v)[:s_list[i]]] = -1

        A_g_v_error[i, k] = np.linalg.norm(A_g@(r_v + ones) - ones , ord = 2)
        #r_v_c[np.argsort(v_2_c)[:s_list[i]]] = -1
        #A_c_v2_error[i] = np.linalg.norm(A_c@(r_v_c + ones) - ones , ord = 2)
        #r_v_t[np.argsort(v_2_t)[:s_list[i]]] = -1
        #A_t_v2_error[i] = np.linalg.norm(A_t@(r_v_t + ones) - ones , ord = 2)


        print(np.sum((r_v + ones) == 0))
        #print(np.sum((r_v_c + ones) == 0))
        print(s_list[i])


c = cp.Variable((1,n))
t = cp.Variable()

#Construct the eigenvectors of a circulant matrix
roots_of_unity = np.exp(2j * np.pi * np.arange(1,n) / n)
eigen_matrix = np.vander(roots_of_unity, n, increasing=True).T #columns are eigenvectors of the circulant matrix

iter_num = int(sys.argv[3])
slems = np.zeros((1,iter_num))
seeds = np.arange(iter_num)
opt_c = np.zeros((iter_num, n))

for o in range(0,iter_num):
    print(o)
    np.random.seed(seeds[o])
    #nonzero_entries = np.random.choice(np.arange(n), size= d, replace=False)
    zero_entries = np.random.choice(np.arange(n), size= n-d, replace=False)
    #zero_entries = np.setdiff1d(np.arange(n), nonzero_entries)
    A = np.zeros((n, len(zero_entries)))
    for i in range(len(zero_entries)):
      A[ zero_entries[i],i] = 1

    #Define the objective function
    objective = cp.Minimize(t)
    ones_vector = cp.Constant(np.ones((1,n)))
    constraints = [c@ones_vector.T ==1 , c@A == 0, c>=0, cp.abs(c@eigen_matrix)<=t] #how can this be unbounded?? (even for c>=0 it happens)


    # Create the problem
    problem = cp.Problem(objective, constraints)



    if True:
      # Solve the problem
      problem.solve(verbose =True)


      # Print the results
      if problem.status == cp.OPTIMAL:
          print("Optimal value:", problem.value)
          #print("Optimal c:", c.value)
          c_r = np.where(c.value < 0.001, 0, c.value)
          c_rounded = np.round(c_r, 3)
          print("Optimal c rounded:", c_rounded)
          print(np.where(c_rounded>0))
          #file_path = 'Circulant Matrix_' + str(n) +'_' + str(d) + '\\' + str(o)+  '_c' + '_' + str(n) + '_' + str(d)+ '_' +  str(np.round(problem.value, 3))  + '.npy'
          #np.save(file_path, c.value)
          slems[0,o] = problem.value
          opt_c[o] = c.value

          print(problem.value)
      else:
          print("Problem is infeasible or unbounded")
          c_r = np.where(c.value < 0.001, 0, c.value)
          c_rounded = np.round(c_r, 3)
          print("Optimal c rounded:", c_rounded)
          print(np.where(c_rounded>0))

          #file_path = 'Circulant Matrix_' + str(n) +'_' + str(d) + '\\' + str(o)+  '_c' + '_' + str(n) + '_' + str(d)+ '_' +  str(np.round(problem.value, 3))  + '.npy'
          #np.save(file_path, c.value)
          slems[0,o] = problem.value
          opt_c[o] = c.value
          print(problem.value)
      """
    indices = np.where(c_rounded>0)
    indices
    C = np.zeros((n,n))
    for i in range(n):
    C[i, :] = np.roll(c.value, i)
    eigenvalues = np.linalg.eigvals(C)
    abs_eigenvalues = np.abs(eigenvalues)
    sorted_eigenvalues = np.sort(abs_eigenvalues)[::-1]
    sorted_eigenvalues
    """

#np.save( 'Circulant Matrix_' + str(n) +'_' + str(d) + '\\' + 'slems.npy', slems)
#np.save('Circulant Matrix_' + str(n) +'_' + str(d) + '\\' + 'opt_c.npy', opt_c)


#Load best optimized c of a Circualnt Matrix
#c= np.load('')
#Or find the best optimal c from the optimization problems run above
c = opt_c[np.argmin(slems)]
A_c = np.zeros((n,n))
for i in range(n):
    A_c[i] = np.roll(c, i)
IF_circ = np.sort(np.abs(np.linalg.eigvals(A_c)))[::-1][1]
#print('IF', IF_circ)
#print('minimum optimized slem', np.min(slems))

#E_Values_c, E_VECTORS_c = np.linalg.eig(A_c)
#v_2 = E_VECTORS[:, 0]
#print(E_VECTORS[:, 0])
#print(np.sort(np.abs(E_Values)))
#print(E_Values_c)




#evals_c = np.abs(E_Values_c)
#print(evals_c)
#print(np.sort(evals_c))
#print(evals_c[7])

#v_2_c = E_VECTORS[:, 7]


# Transition Matrix Optimization With Given Parameters


# Define the size of the matrices
P = cp.Variable((n, n))
ones_vector = cp.Constant(np.ones((1, n)))
X = (1/n)*ones_vector.T@ones_vector
inverted_matrix = np.where(A_g == 0, 1, 0)
constraints = [ cp.multiply(P, inverted_matrix) == 0, P==P.T, P>=0, P@ones_vector.T == ones_vector.T]
# Create the optimization problem

objective = cp.Minimize(cp.norm((P-X) ,2))

prob = cp.Problem(objective, constraints)
prob.solve(verbose = True, solver = cp.SCS)

# Get the optimal value of the variable matrix
if prob.status == cp.OPTIMAL:
  print("Optimal value:", prob.value)
  P_r = np.where(P.value < 0.001, 0, P.value)
  P_rounded = np.round(P_r, 3)
  #file_path =  'Transition Matrices Random/' +  '_P_'+ 'graph_IF_' + str(graph_IF) + '_random_' + str(n) + '_'+ str(d)+ '_' + str(prob.value) + '.npy'
  #np.save(file_path, P.value)



#print("Optimal P:", P.value)
#print("Optimal P rounded:", P_rounded)0.

else:
  print("Problem is infeasible or unbounded")







#load a transition matrix
A_t = P.value


#Transition Matrix IF
IF_t = np.sort(np.abs(np.linalg.eigvals(A_t)))[::-1][1]
print(IF_t)

#E_Values_t, E_VECTORS_t = np.linalg.eig(A_t)
#v_2 = E_VECTORS[:, 0]
#print(E_VECTORS[:, 0])
#print(np.sort(np.abs(E_Values)))
#print(E_Values_t)




# evals_t = np.abs(E_Values_t)
# print(evals_t)
# print(np.sort(evals_t))
# print(evals_t[171])

# v_2_t = E_VECTORS_t[:, 171]


# ev_abs = np.abs(E_Values_t)
# max_index = np.argmax(ev_abs)
# ev_abs_copy = np.copy(ev_abs)
# ev_abs_copy[max_index] = -np.inf

# v_test = E_VECTORS_t[:, np.argmax(ev_abs_copy)]

# print(np.argmax(ev_abs_copy))
# print(v_2_t )
# print(v_test)

A_c_upper_bound = (IF_circ)* np.sqrt((n*s_list)/(n-s_list))
A_t_upper_bound = (IF_t)* np.sqrt((n*s_list)/(n-s_list))

plt.clf()
plt.plot(frac_stragglers, A_g_upper_bound, marker ='+', label ='Upper Bound of Graph')
plt.plot(frac_stragglers, A_t_upper_bound, marker ='+', label ='Upper Bound of Transition Matrix')
plt.plot(frac_stragglers, A_c_upper_bound, marker ='+', label ='Upper Bound of Circulant Matrix')
#plt.plot(frac_stragglers, np.max(A_g_v_error, axis = 1), marker ='^', label = 'max of r vased on v2-v100, Graph')
plt.plot(frac_stragglers, A_g_x_error, marker ='^', label = 'Worst Case Error for Graph (MILP, 300 sec)')
#plt.plot(frac_stragglers, A_g_opt_error, marker ='^', label = 'Optimal Error for Graph (MILP, 300 sec)')
plt.xlabel('Straggler Fraction', fontsize = 14)
plt.ylabel('Fixed Decoding Error', fontsize = 14)
plt.legend(fontsize='large',loc='upper left')
plt.grid(True)
#plt.title('Approximation Error vs Fraction of Workers Straggling', fontsize = 14)
plt.savefig('Plots_V5//Fixed Decoding_tests_with_MILP_(300)_' + str(n) + '_' + str(d)+ '.pdf')

np.save('Plots_V5//A_g_upper_bound_' + str(n)+ '_' + str(d) +  '.npy', A_g_upper_bound)
np.save('Plots_V5//A_t_upper_bound_' + str(n)+ '_' + str(d) +  '.npy', A_t_upper_bound)
np.save('Plots_V5//A_c_upper_bound_' + str(n)+ '_' + str(d) +  '.npy', A_c_upper_bound)
np.save('Plots_V5//A_g_x_error' + str(n)+ '_' + str(d) +  '.npy', A_g_x_error)



# In[ ]:




