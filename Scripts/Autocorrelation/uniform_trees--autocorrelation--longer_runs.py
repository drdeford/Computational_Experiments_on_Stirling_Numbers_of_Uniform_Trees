# importing the libraries 
# make sure that tree_functions_2.py is in the same directory as this notebook

from tree_functions_2 import *

from statsmodels.graphics import tsaplots

# change figure configurations

#%matplotlib inline

import matplotlib

font = {'size':16}

matplotlib.rc('font', **font)

matplotlib.rc('figure', figsize = (7.0, 7.0))

# the order of the tree

n = 25

# complete graph of order n

K_n = nx.complete_graph(n)

# the algorithm for updating tree T based on the graph G

def tree_cycle_walk_cut(T, G):
    
    tempo = 0
    
    tedges = set(T.edges())
    
    T_new = T.copy()
    
    while tempo == 0:
        
        edge = random.choice(list(G.edges()))
        
        if (edge[0], edge[1]) not in tedges and (edge[1], edge[0]) not in tedges:
            
            tempo = 1
            
            T_new.add_edge(edge[0], edge[1])
            
            ncycle = nx.find_cycle(T_new, edge[0])
            
            cutedge = random.choice(tuple(ncycle))
            
            T_new.remove_edge(cutedge[0], cutedge[1])
    
    return T_new


# function for updating the tree based on the complete graph of the same order using maximum degree

def update_tree_max_deg(T, n_iter = 1000, p = 0.1, q = 0.001, r = 0.001, seed = 1764):
    
    random.seed(seed)
    
    G = nx.complete_graph(n)
        
    tgraphs = [T]
        
    max_deg_seq = [get_max_degree(T)]
    
    for i in range(n_iter):

        if random.random() < p:
            
            temp_T = tree_cycle_walk_cut(tgraphs[-1], G)
    
            if get_max_degree(tgraphs[-1]) <= get_max_degree(temp_T):
                
                if random.random() < q:
                    
                    tgraphs.append(temp_T)
                    
                    max_deg_seq.append(get_max_degree(temp_T))
                    
                else:
            
                    tgraphs.append(tgraphs[-1])
            
                    max_deg_seq.append(get_max_degree(tgraphs[-1]))
            
            else:
                
                if random.random() < r:
                    
                    tgraphs.append(temp_T)
                    
                    max_deg_seq.append(get_max_degree(temp_T))
                    
                else:
            
                    tgraphs.append(tgraphs[-1])
            
                    max_deg_seq.append(get_max_degree(tgraphs[-1]))
                    
        else:
            
            tgraphs.append(tgraphs[-1])
            
            max_deg_seq.append(get_max_degree(tgraphs[-1]))
                    
    return [tgraphs[-1], max_deg_seq]

# function for updating the tree based on the complete graph of the same order using degree centrality

def update_tree_deg(T, n_iter = 1000, p = 0.1, q = 0.001, r = 0.001, seed = 1764):
    
    random.seed(seed)
    
    G = nx.complete_graph(n)
        
    tgraphs = [T]
        
    deg_seq = [get_degree_centrality(T)]
    
    for i in range(n_iter):

        if random.random() < p:
            
            temp_T = tree_cycle_walk_cut(tgraphs[-1], G)
    
            if get_degree_centrality(tgraphs[-1]) <= get_degree_centrality(temp_T):
                
                if random.random() < q:
                    
                    tgraphs.append(temp_T)
                    
                    deg_seq.append(get_degree_centrality(temp_T))
                    
                else:
            
                    tgraphs.append(tgraphs[-1])
            
                    deg_seq.append(get_degree_centrality(tgraphs[-1]))
            
            else:
                
                if random.random() < r:
                    
                    tgraphs.append(temp_T)
                    
                    deg_seq.append(get_degree_centrality(temp_T))
                    
                else:
            
                    tgraphs.append(tgraphs[-1])
            
                    deg_seq.append(get_degree_centrality(tgraphs[-1]))
                    
        else:
            
            tgraphs.append(tgraphs[-1])
            
            deg_seq.append(get_degree_centrality(tgraphs[-1]))
                    
    return [tgraphs[-1], deg_seq]


# function for updating the tree based on the complete graph of the same order using closeness centrality

def update_tree_cls(T, n_iter = 1000, p = 0.1, q = 0.001, r = 0.001, seed = 1764):
    
    random.seed(seed)
    
    G = nx.complete_graph(n)
        
    tgraphs = [T]
        
    cls_seq = [get_closeness_centrality(T)]
    
    for i in range(n_iter):

        if random.random() < p:
            
            temp_T = tree_cycle_walk_cut(tgraphs[-1], G)
    
            if get_closeness_centrality(tgraphs[-1]) <= get_closeness_centrality(temp_T):
                
                if random.random() < q:
                    
                    tgraphs.append(temp_T)
                    
                    cls_seq.append(get_closeness_centrality(temp_T))
                    
                else:
            
                    tgraphs.append(tgraphs[-1])
            
                    cls_seq.append(get_closeness_centrality(tgraphs[-1]))
            
            else:
                
                if random.random() < r:
                    
                    tgraphs.append(temp_T)
                    
                    cls_seq.append(get_closeness_centrality(temp_T))
                    
                else:
            
                    tgraphs.append(tgraphs[-1])
            
                    cls_seq.append(get_closeness_centrality(tgraphs[-1]))
                    
        else:
            
            tgraphs.append(tgraphs[-1])
            
            cls_seq.append(get_closeness_centrality(tgraphs[-1]))
                    
    return [tgraphs[-1], cls_seq]

# function for updating the tree based on the complete graph of the same order using betweenness centrality

def update_tree_btw(T, n_iter = 1000, p = 0.1, q = 0.001, r = 0.001, seed = 1764):
    
    random.seed(seed)
    
    G = nx.complete_graph(n)
        
    tgraphs = [T]
        
    btw_seq = [get_betweenness_centrality(T)]
    
    for i in range(n_iter):

        if random.random() < p:
            
            temp_T = tree_cycle_walk_cut(tgraphs[-1], G)
    
            if get_betweenness_centrality(tgraphs[-1]) <= get_betweenness_centrality(temp_T):
                
                if random.random() < q:
                    
                    tgraphs.append(temp_T)
                    
                    btw_seq.append(get_betweenness_centrality(temp_T))
                    
                else:
            
                    tgraphs.append(tgraphs[-1])
            
                    btw_seq.append(get_betweenness_centrality(tgraphs[-1]))
            
            else:
                
                if random.random() < r:
                    
                    tgraphs.append(temp_T)
                    
                    btw_seq.append(get_betweenness_centrality(temp_T))
                    
                else:
            
                    tgraphs.append(tgraphs[-1])
            
                    btw_seq.append(get_betweenness_centrality(tgraphs[-1]))
                    
        else:
            
            tgraphs.append(tgraphs[-1])
            
            btw_seq.append(get_betweenness_centrality(tgraphs[-1]))
                    
    return [tgraphs[-1], btw_seq]

#--

tree_list = []

t = get_spanning_tree_u_w(K_n)
    
tgraph = nx.Graph()
    
tgraph.add_edges_from(t)
    
tgraph = update_tree_max_deg(tgraph, n_iter = 10000000, p = 0.01, q = 0.05, r = 0.01)
    
tree_list.append(tgraph[0])
    
tsaplots.plot_acf(tgraph[1], lags = 10000000, use_vlines = False, fft = False, zero = True, 
                  alpha = 0.05, color = 'red', markersize = 1, markevery = 50,  
                  title = 'Autocorrelation -- Max. Degree')
    
plt.axhline(y = 0, color = 'blue')
    
plt.savefig("./auto_figs/1mdegree1.png")

plt.close()

#--

tree_list = []

t = get_spanning_tree_u_w(K_n)
    
tgraph = nx.Graph()
    
tgraph.add_edges_from(t)
    
tgraph = update_tree_deg(tgraph, n_iter = 10000000, p = 0.01, q = 0.05, r = 0.01)
    
tree_list.append(tgraph[0])
    
tsaplots.plot_acf(tgraph[1], lags = 10000000, use_vlines = False, fft = False, zero = True, 
                  alpha = 0.05, color = 'red', markersize = 1, markevery = 50,  
                  title = 'Autocorrelation --  Degree')
    
plt.axhline(y = 0, color = 'blue')
    
plt.savefig("./auto_figs/1mdegree2.png")

plt.close()

#--

tree_list = []

t = get_spanning_tree_u_w(K_n)
    
tgraph = nx.Graph()
    
tgraph.add_edges_from(t)
    
tgraph = update_tree_cls(tgraph, n_iter = 10000000, p = 0.01, q = 0.05, r = 0.01)
    
tree_list.append(tgraph[0])
    
tsaplots.plot_acf(tgraph[1], lags = 10000000, use_vlines = False, fft = False, zero = True, 
                  alpha = 0.05, color = 'red', markersize = 1, markevery = 50,  
                  title = 'Autocorrelation -- Closeness')
    
plt.axhline(y = 0, color = 'blue')
    
plt.savefig("./auto_figs/1mclose.png")

plt.close()

#--

tree_list = []

t = get_spanning_tree_u_w(K_n)
    
tgraph = nx.Graph()
    
tgraph.add_edges_from(t)
    
tgraph = update_tree_btw(tgraph, n_iter = 10000000, p = 0.01, q = 0.05, r = 0.01)
    
tree_list.append(tgraph[0])
    
tsaplots.plot_acf(tgraph[1], lags = 10000000, use_vlines = False, fft = False, zero = True, 
                  alpha = 0.05, color = 'red', markersize = 1, markevery = 50,  
                  title = 'Autocorrelation -- Betweenness')
    
plt.axhline(y = 0, color = 'blue')
    
plt.savefig("./auto_figs/1mbetween.png")

plt.close()
