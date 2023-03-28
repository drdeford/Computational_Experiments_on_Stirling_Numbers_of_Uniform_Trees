# importing the libraries 
# make sure that tree_functions_2.py is in the same directory as this notebook

from tree_functions_2 import * 

import time

import matplotlib.pyplot as plt

#--

def avg_time_stir_sim(n = 25, m = 20000, seed = 42, I = 10):
    
    Time = []
    
    for i in range(I):
        
        st = time.time()

        simulate_stirling(n, m, seed)

        Time.append(time.time() - st)
        
    return np.mean(Time)

#--

def avg_time_stir_exact(n = 25, seed = 42, I = 10):
    
    Time = []
    
    T = nx.random_tree(n, seed)
    
    for i in range(I):
        
        st = time.time()
    
        Stir = get_stirling_trees(T, n)

        Time.append(time.time() - st)
        
    return np.mean(Time)

#--

Avg_Time_Sim = []

Avg_Time_Exact = []

f = open("./Timing_Outputs10k_2.txt",'w')

for i in range(5,31):
    
    Avg_Time_Sim.append(avg_time_stir_sim(n = i, m = 10000, seed = 42))

    print(f"Average simulation time for trees of size {i} is {Avg_Time_Sim[-1]}")

    f.write(f"Average simulation time for trees of size {i} is {Avg_Time_Sim[-1]}\n\n")
    
    Avg_Time_Exact.append(avg_time_stir_exact(n = i, seed = 42))

    print(f"Average exact computation time for trees of size {i} is {Avg_Time_Exact[-1]}")
    
    f.write(f"Average exact computation time for trees of size {i} is {Avg_Time_Exact[-1]}\n\n")

f.close()

print(Avg_Time_Sim)

print(Avg_Time_Exact)

plt.plot(range(5,31), Avg_Time_Sim, 'g*', label='Simulation')

plt.plot(range(5,31), Avg_Time_Exact, 'bd', label='Exact')

plt.ylabel('Average Time (s)')

plt.xlabel('Size of Tree (n)')

plt.legend()

plt.savefig('./comparison_10k.png')

plt.close()

#--

Avg_Time_Sim = []

Avg_Time_Exact = []

f = open("./Timing_Outputs20k_2.txt", 'w')

for i in range(5,31):
    
    Avg_Time_Sim.append(avg_time_stir_sim(n = i, m = 20000, seed = 42))

    print(f"Average simulation time for trees of size {i} is {Avg_Time_Sim[-1]}")

    f.write(f"Average simulation time for trees of size {i} is {Avg_Time_Sim[-1]}\n\n")
    
    Avg_Time_Exact.append(avg_time_stir_exact(n = i, seed = 42))

    print(f"Average exact computation time for trees of size {i} is {Avg_Time_Exact[-1]}")
    
    f.write(f"Average exact computation time for trees of size {i} is {Avg_Time_Exact[-1]}\n\n")

f.close()

print(Avg_Time_Sim)

print(Avg_Time_Exact)

plt.plot(range(5,31), Avg_Time_Sim, 'g*', label='Simulation')

plt.plot(range(5,31), Avg_Time_Exact, 'bd', label='Exact')

plt.ylabel('Average Time (s)')

plt.xlabel('Size of Tree (n)')

plt.legend()

plt.savefig('./comparison_20k.png')

plt.close()
