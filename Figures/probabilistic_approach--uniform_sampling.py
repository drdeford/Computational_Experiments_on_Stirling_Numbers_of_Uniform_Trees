# importing the libraries 
# make sure that tree_functions_2.py is in the same directory as this notebook

from tree_functions_2 import * 

import time

import csv

def uniform_stirling_diff(n, m):

    K_n = nx.complete_graph(n)
    
    T = nx.to_networkx_graph(get_spanning_tree_u_w(K_n))
    
    Stir = get_stirling_trees(T, n)
    
    A = sparse.spmatrix.toarray(my_incidence_matrix(T))
    
    one = np.matrix(np.ones(n - 1, dtype = int)).transpose()
    
    Results = []

    for k in range(math.ceil(n/2), n - 1):
            
        stir = []
        
        for j in range(m):
            
            y = np.zeros(n - 1, dtype = int)

            pos = random.sample(range(n - 1), k - 1)

            for i in range(n):
                
                if i in pos:
                    
                    y[i] = 1

            y = np.matrix(y).transpose()
               
            x = np.matmul(A, (one - y)).transpose()

            a = sum([x[0, i] == 1 for i in range(n)])

            stir.append(a == 2*(n - k))
 
        Results.append([n, k, Stir[-(n - k + 1)] - sum(stir) * special.comb(n - 1, k - 1, exact = True)/m]) #Absolute Error
    
    return Results

random.seed(42)

results = []

for n in range(7, 20):

    print(n)

    results.append([])
    
    for i in range(100):

        results[-1].append(uniform_stirling_diff(n, m = 10000)) 

    print(results)
    
    with open(f"./auto_csv/uniform_comparison_{n}.csv",'w') as f:
        
        wr = csv.writer(f)
        
        wr.writerows(results[-1])


with open("./auto_csv/uniform_comparison_all.csv",'w') as f:
    
    wr = csv.writer(f)
    
    wr.writerows(results)
