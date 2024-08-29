#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cvxpy as cp
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os


# In[2]:


n = int(sys.argv[1])

d = int(sys.argv[2])
random_seed = 42
ones = np.ones((n,1))
#Define a graph

#G = nx.random_regular_expander_graph(n, d)
#G = nx.margulis_gabber_galil_graph(n)
G = nx.random_regular_graph(d, n, seed = random_seed)


# In[3]:


A_g = (1/d)*nx.to_numpy_array(G)


# In[4]:


s_list = d* np.arange(1, 6)
opt_err_array = np.zeros((len(s_list)))
fixed_err_array = np.zeros((len(s_list)))

for j in range(len(s_list)):
    
        s = s_list[j]
        x = np.nonzero(A_g[0,:])[0]
        parts = [0]
        list1 =  list(range(n))
        result_list = [item for item in list1 if item not in parts]
        tmp1 = []
        while(len(x) < s):
                tmp = 0
                for i in result_list:
                    c = np.intersect1d(np.nonzero(A_g[i,:]), x).size
                    #print('c:',c)
                    #print("tmp:", tmp)
                    if c > tmp: 
                        f = i
                        tmp = c
                if tmp == 0:
                    f = result_list[0]
                    tmp1 = x
                    x = np.union1d(x, np.nonzero(A_g[f,:])[0])
                    parts.append(f)
                    result_list = [item for item in list1 if item not in parts]

                else:
                    tmp1 = x
                    x = np.union1d(x, np.nonzero(A_g[f,:])[0])
                    print(x)
                    parts.append(f)
                    result_list = [item for item in list1 if item not in parts]


                if len(x) > s:
                    print("shoot")
                    print(len(x))
                    diff = np.setdiff1d(np.nonzero(A_g[f,:])[0], tmp1)
                    x = np.union1d(tmp1, diff[0:s-len(tmp1)])
                    print(len(x))

        print(n)        
        u = (s) /( n -s) * np.ones((n,1))
        u[x] = -1
        r = u + np.ones((n,1))

        A_g_F = A_g[:, np.setdiff1d(np.arange(0,n),x)]

        fixed =  np.linalg.norm(A_g@r - ones , ord = 2)

        opt =  np.sqrt(n-ones.T@A_g_F@ np.transpose(np.linalg.inv(np.transpose(A_g_F)@A_g_F))@A_g_F.T@ones)
        
        opt_err_array[j] = opt
        fixed_err_array[j] = fixed


# In[5]:


evals = np.sort(np.abs(np.linalg.eigvals(A_g)))[::-1]
#print(evals)
IF_g = evals[1]/evals[0]


# In[6]:


bound = (IF_g)* np.sqrt((n*s_list)/(n-s_list))


# In[7]:


bound


# In[8]:


opt_err_array


# In[9]:


fixed_err_array


# In[10]:


P = cp.Variable((n, n))
ones_vector = cp.Constant(np.ones((1, n)))
X = (1/n)*ones_vector.T@ones_vector
inverted_matrix = np.where(A_g == 0, 1, 0)
constraints = [ cp.multiply(P, inverted_matrix) == 0, P==P.T, P>=0, P@ones_vector.T == ones_vector.T]
# Create the optimization problem

objective = cp.Minimize(cp.norm((P-X) ,2))

prob = cp.Problem(objective, constraints)
prob.solve(verbose = False, solver = cp.SCS)

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


# In[11]:


A_t = np.round(A_t, 3)


# In[12]:


A_t


# In[13]:


A_t_opt_err_array = np.zeros((len(s_list)))
A_t_fixed_err_array = np.zeros((len(s_list)))

for j in range(len(s_list)):
        print(j)
        s = s_list[j]
        x = np.nonzero(A_t[0,:])[0]
        parts = [0]
        list1 =  list(range(n))
        result_list = [item for item in list1 if item not in parts]
        tmp1 = []
        while(len(x) < s):
                tmp = 0
                for i in result_list:
                    c = np.intersect1d(np.nonzero(A_t[i,:]), x).size
                    #print('c:',c)
                    #print("tmp:", tmp)
                    if c > tmp: 
                        f = i
                        tmp = c
                if tmp == 0:
                    f = result_list[0]
                    tmp1 = x
                    x = np.union1d(x, np.nonzero(A_t[f,:])[0])
                    parts.append(f)
                    result_list = [item for item in list1 if item not in parts]

                else:
                    tmp1 = x
                    x = np.union1d(x, np.nonzero(A_t[f,:])[0])
                    print(x)
                    parts.append(f)
                    result_list = [item for item in list1 if item not in parts]


                if len(x) > s:
                    print("shoot")
                    print(len(x))
                    diff = np.setdiff1d(np.nonzero(A_t[f,:])[0], tmp1)
                    x = np.union1d(tmp1, diff[0:s-len(tmp1)])
                    print(len(x))

        print(x)
        print(n)        
        u = (s) /( n -s) * np.ones((n,1))
        u[x] = -1
        r = u + np.ones((n,1))

        A_t_F = A_t[:, np.setdiff1d(np.arange(0,n),x)]

        fixed =  np.linalg.norm(A_t@r - ones , ord = 2)

        opt =  np.sqrt(n-ones.T@A_t_F@ np.transpose(np.linalg.inv(np.transpose(A_t_F)@A_t_F))@A_t_F.T@ones)
        
        A_t_opt_err_array[j] = opt
        A_t_fixed_err_array[j] = fixed


# In[14]:


A_t_opt_err_array


# In[15]:


A_t_fixed_err_array


# In[16]:


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
      problem.solve(verbose = False)


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


# In[17]:


A_c_opt_err_array = np.zeros((len(s_list)))
A_c_fixed_err_array = np.zeros((len(s_list)))

for j in range(len(s_list)):
        print(j)
        s = s_list[j]
        x = np.nonzero(A_c[0,:])[0]
        parts = [0]
        list1 =  list(range(n))
        result_list = [item for item in list1 if item not in parts]
        tmp1 = []
        while(len(x) < s):
                tmp = 0
                for i in result_list:
                    c = np.intersect1d(np.nonzero(A_c[i,:]), x).size
                    #print('c:',c)
                    #print("tmp:", tmp)
                    if c > tmp: 
                        f = i
                        tmp = c
                if tmp == 0:
                    f = result_list[0]
                    tmp1 = x
                    x = np.union1d(x, np.nonzero(A_c[f,:])[0])
                    parts.append(f)
                    result_list = [item for item in list1 if item not in parts]

                else:
                    tmp1 = x
                    x = np.union1d(x, np.nonzero(A_c[f,:])[0])
                    print(x)
                    parts.append(f)
                    result_list = [item for item in list1 if item not in parts]


                if len(x) > s:
                    print("shoot")
                    print(len(x))
                    diff = np.setdiff1d(np.nonzero(A_c[f,:])[0], tmp1)
                    x = np.union1d(tmp1, diff[0:s-len(tmp1)])
                    print(len(x))

        print(x)
        print(n)        
        u = (s) /( n -s) * np.ones((n,1))
        u[x] = -1
        r = u + np.ones((n,1))

        A_c_F = A_c[:, np.setdiff1d(np.arange(0,n),x)]

        fixed =  np.linalg.norm(A_c@r - ones , ord = 2)

        opt =  np.sqrt(n-ones.T@A_c_F@ np.transpose(np.linalg.inv(np.transpose(A_c_F)@A_c_F))@A_c_F.T@ones)
        
        A_c_opt_err_array[j] = opt
        A_c_fixed_err_array[j] = fixed


# In[18]:


frac_stragglers = s_list/n


# In[19]:


A_c_fixed_err_array


# In[20]:


A_c_opt_err_array


# In[21]:


A_c_bound = (IF_circ)* np.sqrt((n*s_list)/(n-s_list))


# In[22]:


IF_t = np.sort(np.abs(np.linalg.eigvals(A_t)))[::-1][1]
A_t_bound = (IF_t)* np.sqrt((n*s_list)/(n-s_list))


# In[33]:


plt.plot(frac_stragglers, A_c_opt_err_array,linestyle = '-', color = 'green', marker ='^', label = 'Opt. error for circulant')
plt.plot(frac_stragglers, A_t_opt_err_array,linestyle = '-', color = 'orange', marker ='+', label = 'Opt. error for transition')
plt.plot(frac_stragglers, opt_err_array,linestyle = '-', color = 'blue', marker ='d', label = 'Opt. error for graph')


plt.plot(frac_stragglers,  A_c_bound, linestyle = ':', color = 'green', marker  ='s', label = 'Upper bound for circulant')
plt.plot(frac_stragglers, A_t_bound,linestyle = ':', color = 'orange', marker ='o', label = 'Upper bound for transition')
plt.plot(frac_stragglers, bound ,linestyle = ':', color = 'blue', marker ='x', label = 'Upper bound for graph')



# plt.plot(frac_stragglers,  A_c_fixed_err_array , linestyle = ':', color = 'green', marker  ='s', label = 'Fixed error for circulant')
# plt.plot(frac_stragglers, A_t_fixed_err_array,linestyle = ':', color = 'orange', marker ='o', label = 'Fixed error for transition')
# plt.plot(frac_stragglers, fixed_err_array ,linestyle = ':', color = 'blue', marker ='x', label = 'Fixed error for graph')

plt.xlabel('Straggler Fraction', fontsize = 14)
plt.ylabel('Error', fontsize = 14)
plt.legend(fontsize='large',loc='upper left')
plt.rc('text', usetex=True)
plt.grid(True)
#plt.title('Approximation Error vs Fraction of Workers Straggling', fontsize = 14)
plt.savefig('Plots_V6//V4_Opt. Decoding_tests_ALG_with_bound_' + str(n) + '_' + str(d)+ '.pdf')


# In[ ]:




