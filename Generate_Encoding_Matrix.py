#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cvxpy as cp
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os
import sys


# In[2]:


# #input parameters (n, d) 
n = int(sys.argv[1])
d = int(sys.argv[2])
iter_num = int(sys.argv[3])


# In[3]:


#Generate a random regular a graph

G = nx.random_regular_graph(d, n) 

#Adjacency matrix of the graph

A_g = nx.to_numpy_array(G)





# In[4]:


graph_evals = np.sort(np.abs(np.linalg.eigvals(A_g)))[::-1]
#print(evals)
IF_g = graph_evals[1]/graph_evals[0]
#print("Improvement Factor of Graph", IF_g)


# In[5]:


#Circulant Matrix Optimization with the given parameters 


c = cp.Variable((1,n))
t = cp.Variable()

#Construct the eigenvectors of a circulant matrix
roots_of_unity = np.exp(2j * np.pi * np.arange(1,n) / n)
eigen_matrix = np.vander(roots_of_unity, n, increasing=True).T #columns are eigenvectors of the circulant matrix


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


# In[6]:


#Load best optimized c of a Circualnt Matrix 
#c= np.load('')
#Or find the best optimal c from the optimization problems run above
c = opt_c[np.argmin(slems)]
A_c = np.zeros((n,n))
for i in range(n):
    A_c[i] = np.roll(c, i)
IF_circ = np.sort(np.abs(np.linalg.eigvals(A_c)))[::-1][1]
#print('Improvement Factor of Circulant Matrix', IF_circ)
#print('minimum optimized slem', np.min(slems))


# In[7]:


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




          
          


# In[8]:


#load a transition matrix
A_t = P.value 


#Transition Matrix IF
IF_t = np.sort(np.abs(np.linalg.eigvals(A_t)))[::-1][1]


# In[13]:


print('Improvement Factor of Graph', IF_g)
print('Improvement Factor of Circulant Matrix', IF_circ)
print('Improvement Factor of Transition Matrix', IF_t)


# In[14]:


np.save('Adjacency Matrix of Graph_' + str(n) + '_' + str(d) + '.npy', A_g)
np.save('Cirucalnt Matrix_' + str(n) + '_' + str(d) + '.npy', A_c)
np.save('Transition Matrix_' + str(n) + '_' + str(d) + '.npy', A_t)


# In[ ]:




