# loading packages

import networkx as nx

import numpy as np

import scipy as sp

import pandas as pd

import seaborn as sns

import random

from random import sample

from random import choice

import matplotlib.pyplot as plt

from scipy import sparse

from scipy import special

from sympy import symbols

from sympy import parse_expr

from sympy import simplify

from sympy import expand

from sympy import degree_list

from sympy import degree

from sympy import *

from itertools import combinations

import math

from networkx import isomorphism

# generates a balanced tentacle tree with n tentacles
# and 2n + 1 vertices

def gen_tentacle_tree(n):
    
    tentacle_tree = nx.Graph()

    tentacle_tree.add_node(0)

    alphabet = 'abcdefghijklmnopqrstuvwxyz'

    for i in range(n):

        tentacle_tree.add_edge(0, alphabet[i])

        tentacle_tree.add_edge(alphabet[i] + '1', alphabet[i])

    return tentacle_tree

# generates a balanced stem tree on 2n + 1 vertices

def gen_stem_tree(n):
    
    stem_tree = nx.Graph()

    stem_tree.add_node(0)
    
    alphabet = 'abcdefghijklmnopqrstuvwxyz'

    for i in range(n):

        stem_tree.add_edge(0, alphabet[i] + '0')

    for i in range(n):

        stem_tree.add_edge('a' + str(i), 'a' + str(i + 1))

    return stem_tree

# generates a balanced dumbell tree with 2n+2 vertices 

def gen_dumbell_tree(n):

    dumbell_tree = nx.Graph()
    
    dumbell_tree.add_node(0)

    alphabet = 'abcdefghijklmnopqrstuvwxyz'

    for i in range(n):

        dumbell_tree.add_edge(0, alphabet[i] + '0')

    for i in range(n):
    
        dumbell_tree.add_edge('0hat', alphabet[i] + '0hat')
    
    dumbell_tree.add_edge(0, '0hat')

    return dumbell_tree

# deletion-inclusion algorithm

def deletion_inclusion_trees(G):
    
    H = G.copy()
    
    K = G.copy()
    
    E = G.edges()
    
    e = sample(E, 1)
    
    u, v = tuple(*e)
    
    H.remove_edge(u, v)
    
    incident_edges = list(nx.edge_boundary(K, *e))
    
    K.remove_edges_from(incident_edges)
    
    K = nx.contracted_edge(K, *e, self_loops = False)
    
    return H, K

# deletion-inclusion algorithm for a tree

def deletion_inclusion_alg_trees(G):
    
    Added = [G]
    
    Removed = []
    
    while len(Added) != 0:
        
        for A in Added:
        
            P, Q = deletion_inclusion_trees(A)
            
            Added.remove(A)
            
            if P.size() == 0:
            
                Removed.append(P)
            
            else:
            
                Added.append(P)
            
            if Q.size() == 0:
            
                Removed.append(Q)
            
            else:
            
                Added.append(Q)
    
    return Removed

# returns the list of Stirling numbers for a tree 

def get_stirling_trees(T, n): 

    #n = len(T) #I think this is right?#changed to parameter
    
    m = int(np.ceil((n + 1) / 2))
    
    L = deletion_inclusion_alg_trees(T)
    
    C = [L[i].order() for i in range(len(L))]
    
    unique, counts = np.unique(C, return_counts = True)
    
    truecounts = np.zeros(m, dtype = int)
    
    for index, i in enumerate(unique):
        
        if i in unique:
        
            if n %2 != 0:
            
                truecounts[i - m] = counts[index]
            
            else:
            
                truecounts[i - m + 1] = counts[index]
    
    return list(truecounts)

#--

def get_closeness_seq(G):
    
    return sorted([x[1] for x in nx.closeness_centrality(G).items()])

#--

def get_max_closeness(G):
    
    return max([x[1] for x in nx.closeness_centrality(G).items()])

#--

def get_closeness_centrality(G):

    closeness = get_closeness_seq(G)
    
    n = G.order()
    
    return sum([(closeness[-1] - closeness[i])/((n - 1)*(n - 2)/(2*n - 3)) for i in range(n)])

#--

def get_betweenness_seq(G):
    
    return sorted([x[1] for x in nx.betweenness_centrality(G).items()])

#--

def get_max_betweenness(G):
    
    return max([x[1] for x in nx.betweenness_centrality(G).items()])

#--

def get_betweenness_centrality(G):
   
    betweenness = get_betweenness_seq(G)
    
    n = G.order()
    
    return sum([(betweenness[-1] - betweenness[i])/(n - 1) for i in range(n)])	

#--

def get_degree_seq(G):
    
    return sorted([G.degree(v) for v in G.nodes()])

#--

def get_leaf_number(G):

    return sum([G.degree(v) == 1 for v in G.nodes()])

#--

def get_max_degree(G):
    
    return max([G.degree(v) for v in G.nodes()])

#--

def get_degree_centrality(G):
   
    degrees = get_degree_seq(G)
    
    n = G.order()

    return sum([(degrees[-1] - degrees[i]) / (n**2 - 3*n + 2) for i in range(n)])

#--

def get_closeness_seq(G):
    
    return sorted([x[1] for x in nx.closeness_centrality(G).items()])

#--

def get_max_closeness(G):
    
    return max([x[1] for x in nx.closeness_centrality(G).items()])

#--

def get_closeness_centrality(G):
   
    closeness = get_closeness_seq(G)
    
    n = G.order()
    
    return sum([(closeness[-1] - closeness[i])/((n - 1)*(n - 2)/(2*n - 3)) for i in range(n)])

#--

def get_betweenness_seq(G):
    
    return sorted([x[1] for x in nx.betweenness_centrality(G).items()])

#--

def get_max_betweenness(G):
    
    return max([x[1] for x in nx.betweenness_centrality(G).items()])

#--

def get_betweenness_centrality(G):
   
    betweenness = get_betweenness_seq(G)
    
    n = G.order()
    
    return sum([(betweenness[-1] - betweenness[i])/(n - 1) for i in range(n)])

#--

def step_to_star(tree):
    
    nt = tree.copy()
    
    cc_dict = nx.closeness_centrality(nt)
    
    cc_list = [(key, cc_dict[key]) for key in cc_dict]
    
    cc_list = sorted(cc_list, key = lambda x: x[1])
    
    cc_list.reverse()
    
    #print(cc_list)
    
    leaves = []
    
    for n in nt.nodes():
    
        if nt.degree[n] == 1:
        
            #print(n)
            
            leaves.append((n, nx.shortest_path_length(nt, n, cc_list[0][0])))
                          
    leaves = sorted(leaves, key = lambda x: x[1])
    
    leaves.reverse()
    
    #print(leaves)
    
    nt.remove_node(leaves[0][0])
    
    nt.add_edge(cc_list[0][0], leaves[0][0])
                          
    return nt
  
#--    
    
def step_to_path(tree):
    
    nt = tree.copy()
    
    cc_dict = nx.closeness_centrality(nt)
    
    cc_list = [(key, cc_dict[key]) for key in cc_dict]
    
    cc_list = sorted(cc_list, key = lambda x: x[1])
    
    cc_list.reverse()
    
    path_dict = dict(nx.shortest_path_length(nt))  
    
    #print(nt.nodes())
    
    #print(path_dict)
    
    diam = (0, 0, 0)
    
    for n in nt.nodes():
        
        for m in nt.nodes():
        
            if n != m:
            
                dist = path_dict[n][m]
                
                if dist > diam[2]:
                
                    diam = (n, m, dist)
                

    leaves = []
    
    for n in nt.nodes():
    
        if nt.degree[n] == 1:
        
            leaves.append((n, nx.shortest_path_length(nt, n, cc_list[0][0])))
                          
    leaves = sorted(leaves, key = lambda x: x[1])
    
    nt.remove_node(leaves[0][0])
    
    s = sample([0, 1], 1)[0]
    
    check_node = diam[s]
    
    if leaves[0][0] == check_node:
        
        nt.add_edge(diam[1 - s], leaves[0][0])
    
    else:
    
        nt.add_edge(check_node, leaves[0][0])
                             
    return nt

#--

def cent_score(T):
    
    score_dict = dict()
    
    for node in T.nodes():
    
        score_dict[node] = .1*(len(T) - len(list(T.neighbors(node))))
        
    return score_dict

#--

def quad_score(T):
    
    score_dict = dict()

    for node in T.nodes():
    
        score_dict[node] = (len(list(T.neighbors(node)))**(2.4)) #3/2 worked for just the star #2.4 and a leaf bonus of 2 worked for almost everything
        
        if len(list(T.neighbors(node))) == 1:
        
            score_dict[node] += 2.5
            
        if len(list(T.neighbors(node))) == len(T.nodes) - 1:
            
            score_dict[node] += 1.5
           
    return score_dict

#--

def nodes_score(T):
    
    score_dict = dict()

    for node in T.nodes():
    
        score_dict[node] = (len(list(T.neighbors(node)))**(7/4))  
        
        #if len(list(T.neighbors(node))) == 1:
        
        #    score_dict[node] += 2
        
    return score_dict

#--

def path_score(T):
    
    score_dict = dict()

    for node in T.nodes():
    
        if len(list(T.neighbors(node))) == 1:
        
            score_dict[node] = 100
        
        else:
        
            score_dict[node] = 1
        
    return score_dict

#--

def reverse_engineered_score(T):

    score_dict = dict()
    
    for node in T.nodes():
    
        if len(list(T.neighbors(node))) == 1:
        
            score_dict[node] = 100
        
        else:
        
            score_dict[node] = 10*len(list(T.neighbors(node)))
        
    return score_dict    

#--

def pref_graph(n, score_fun):
    
    T = nx.Graph()
    
    T.add_edge(0, 1)
    
    for i in range(2, n):
        
        D = score_fun(T)
        
        score_list = [D[node] for node in T.nodes()]
        
        norm_list = [x/sum(score_list) for x in score_list]
        
        norm_list = np.cumsum(norm_list)
        
        #print(norm_list)
        
        r = random.random()
        
        temp = 0
        
        for j in range(len(norm_list) - 1):
        
            if norm_list[j] < r and norm_list[j + 1] > r:
            
                temp = j + 1
        
        T.add_edge(i, temp)
        
    return T

#--

def get_path_stirling_check(n):
    
    return (sp.special.binom(n - 1, 2) - (n - 2))/(sp.special.binom(n, 2))

#--

def get_semipath_stirling_check(n):
    
    return (sp.special.binom(n - 1, 2) - (n - 1))/(sp.special.binom(n, 2))

#--

def get_star_stirling_check(n):
    
    return 0

#--

def get_semistar_stirling_check(n):
    
    return (n - 3)/(sp.special.binom(n, 2))

#--

def get_closeness_seq_check(G):
    
    return sorted([x[1] for x in nx.closeness_centrality(G).items()])

#--

def get_max_closeness_check(G):
    
    return max([x[1] for x in nx.closeness_centrality(G).items()])

#--

def get_closeness_centrality_check(G):
   
    closeness = get_closeness_seq_check(G)
    
    n = G.order()
    
    return sum([(closeness[-1] - closeness[i]) for i in range(n)])

#--

def get_path_closeness_even_check(n):
    
    return (4*(n - 1)/n - sum([2*(n - 1)/((n - k)**2 + (k - 1)**2 + n - 1) for k in range(1, n + 1)]))

#--

def get_path_closeness_odd_check(n):
    
    return (4*n/(n + 1) - sum([2*(n - 1)/((n - k)**2 + (k - 1)**2 + n - 1) for k in range(1, n + 1)]))

#--

def get_semipath_closeness_even_check(n):
    
    return (4*(n - 1)**2/(n**2 - 2*n + 4) - 2*sum([(n - 1)/((n - k)**2 + (k - 1)**2 + 1) for k in range(1, n//2)]) - 2*sum([(n - 1)/((n - k)**2 + (k + 1)**2 - 2*n + 1) for k in range(n//2 + 1, n)]) - 4*(n - 1)/(n**2 + 2*n - 4))

#--

def get_semipath_closeness_odd_check(n):
    
    return (4*(n - 1)**2/(n**2 - 2*n + 5) - 2*sum([(n - 1)/((n - k)**2 + (k - 1)**2) for k in range(1, (n - 1)//2)]) - 2*sum([(n - 1)/((n - k)**2 + (k + 1)**2 - 2*n + 2) for k in range((n + 1)//2, n)]) - 4*(n - 1)/(n**2 + 2*n - 3))

#--

def get_star_closeness_check(n):
    
    return (n - 1)*(n - 2)/(2*n - 3)

#--

def get_semistar_closeness_check(n):
    
    return (3*n**3 - 14*n**2 + 17*n - 12)/(6*n*(n - 2))

#--

def get_betweenness_seq_check(G):
    
    return sorted([x[1] for x in nx.betweenness_centrality(G).items()])

#--

def get_betweenness_centrality_check(G):
   
    betweenness = get_betweenness_seq(G)
    
    n = G.order()
    
    return sum([(betweenness[-1] - betweenness[i]) for i in range(n)])

#--

def get_max_betweenness_check(G):
    
    return max([x[1] for x in nx.betweenness_centrality(G).items()])

#--

def get_path_betweenness_even_check(n):
    
    return n*(n + 2)/(6*(n - 1))

#--

def get_path_betweenness_odd_check(n):
    
    return n*(n + 1)/(6*(n - 2))

#--

def get_semipath_betweenness_even_check(n):
    
    return (n**2 + 11*n - 6)/(6*(n - 1))

#--

def get_semipath_betweenness_odd_check(n):
    
    return (n**3 + 9*n**2 - 31*n + 9)/(6*(n - 1)*(n - 2))

#--

def get_star_betweenness_check(n):
    
    return (n - 1)

#--

def get_semistar_betweenness_check(n):
    
    return (n**3 - 4*n**2 + n + 4)/((n - 1)*(n - 2))

#--

def find_poly(T, r):
    
    Q = nx.Graph()
            
    Q.add_nodes_from(T.nodes())
            
    Q.add_edges_from(T.edges())
    
    #Q = T.copy()
    
    Q.remove_node(r)
    
    S = [Q.subgraph(c).copy() for c in nx.connected_components(Q)]
      
    polys = []
    
    for n in T.neighbors(r):
    
        for s in S:
        
            if n in list(s.nodes()):
                
                parent_dict = dict()
                
                polys.append('(')
            
                s.add_edge(r, n)
                
                visited = [r, n]
                
                not_visited = list(s.nodes())
                
                not_visited.remove(r)
                
                not_visited.remove(n)
                
                current = n
                
                while not_visited != []:
                    
                    downs = set(not_visited).intersection(set(s.neighbors(current)))
                    
                    if downs == set():
                
                        polys[-1] = polys[-1] + ')'
                        
                        current = parent_dict[current]
                        
                        """
                        if len(not_visited) >= 2:
                            
                            current = not_visited[-2]
                            
                        else:
                            
                            current = not_visited[0]
                        """
                    
                    else:
                        
                        future = choice(list(downs))
                        
                        visited.append(future)
                        
                        not_visited.remove(future)
                        
                        parent_dict[future] = current
                        
                        current = future
                        
                        polys[-1] = polys[-1] + '('
                
                closes = polys[-1].count('(') - polys[-1].count(')')
                
                for temp in range(closes):
                
                    polys[-1] = polys[-1] + ')'

    return polys

#--

def process_Dyck(polys):
    
    D = []
    
    for p in polys:
        
        D.append('')
        
        if len(p) == 2:
            
            D[-1] = '(x)'
        
        else:
            
            for c in range(len(p) - 1): 
            
                if p[c] == '(' and p[c + 1] == '(':
                
                    D[-1] = D[-1] + '('
                
                if p[c] == '(' and p[c + 1] == ')':
                
                    D[-1] = D[-1] + '(x'
                
                if p[c] == ')' and p[c + 1] == '(': ###this is what we needed to change 
                    
                    if p[c-1] == ')':
                    
                        D[-1] = D[-1] + '+y)*'
                        
                    else:
                        
                        D[-1] = D[-1] + ')*' ###up to here
                
                if p[c] == ')' and p[c + 1] == ')':
                
                    if p[c-1] == '(':
                        
                        D[-1] = D[-1] + ')'
                    
                    else:
                        
                        D[-1] = D[-1] + '+y)'

        
            D[-1] = D[-1] + '+y)'
        
    
    E = ''

    for d in D:
    
        E = E + '*' + d
        
    E = E + '+y'
    
    E = E[1:]
    
    return expand(parse_expr(E))    

#--

def find_poly_unrooted(T):
    
    P = 1
    
    for n in T.nodes():
        
        if T.degree(n) == 1:
            
            H = nx.Graph()
            
            H.add_nodes_from(T.nodes())
            
            H.add_edges_from(T.edges())
    
            u = [m for m in H.neighbors(n)][0]
        
            H.remove_node(n)
    
            P = P*process_Dyck(find_poly(H, u))
    
    return expand(P)

#--

def find_factored_poly_unrooted(T):
    
    P = 1
    
    for n in T.nodes():
        
        if T.degree(n) == 1:
            
            H = nx.Graph()
            
            H.add_nodes_from(T.nodes())
            
            H.add_edges_from(T.edges())
    
            u = [m for m in H.neighbors(n)][0]
        
            H.remove_node(n)
    
            P = P*process_Dyck(find_poly(H, u))
    
    return P

#--

def get_degree_list(pol):
    
    degree_list = []
    
    poly_terms = pol.as_terms()[0]
    
    k = len(poly_terms)
    
    for i in range(k):
        
        degree_list.append([poly_terms[i][1][1][0], poly_terms[i][1][1][1], poly_terms[i][1][0][0]])
        
    return degree_list

#--

def get_matrix_list(tree_list):

    matrix_list = []

    for i in range(len(tree_list)):
        
        df = []
        
        df = pd.DataFrame(get_degree_list(expand(find_factored_poly_unrooted(tree_list[i]))))

        df = df.sort_values(by = [1])

        df = df.sort_values(by = [0], ascending = False)

        df = df.reset_index().iloc[:, 1:4]

        matrix_list.append(np.matrix(df, dtype = 'int'))
    
    return matrix_list

#--

def get_poly_degree_list(tree_list):
    
    x, y = symbols('x y')
    
    poly_list = []
    
    for i in range(len(tree_list)):
        
        poly_list.append(degree(expand(find_factored_poly_unrooted(tree_list[i])), gen = x))
        
    return list(np.unique(np.array(poly_list)))

#--

def get_total_degree_based(tree_list):
    
    matrix_list = []
    
    matrix_list = get_matrix_list(tree_list)
    
    poly_list = []
    
    poly_list = get_poly_degree_list(tree_list)
    
    tree_list_presorted = []
    
    index_sorted = []
    
    for i in range(len(poly_list)):
        
        k = poly_list[i]
                    
        for j in range(len(matrix_list)):
        
            M = matrix_list[j]
    
            if M[0, 0] == k:
            
                tree_list_presorted.append(M) 
                
                index_sorted.append(j)
                
    i = 0
    
    j = 1
    
    L = len(tree_list_presorted) - 1
    
    while i <= L & j < L:
        
        M = tree_list_presorted[i].copy()
           
        N = tree_list_presorted[j].copy()
        
        if M[0, 0] == N[0, 0]:

            l = min(N.shape[0], M.shape[0]) - 1
            
            n = 0
            
            while n <= l:
                    
                if N[n, 0] - M[n, 0] < 0:
                            
                    j = j + 1
                            
                    break
                        
                elif N[n, 0] - M[n, 0] > 0:
                            
                    tree_list_presorted[i] = N
           
                    tree_list_presorted[j] = M
            
                    index_sorted[i] = j
                
                    index_sorted[j] = i
                            
                    i = 0
                    
                    j = 1
                            
                    break
                            
                elif N[n, 0] - M[n, 0] == 0 & N[n, 1] - M[n, 1] > 0:
                            
                    j = j + 1
                                
                    break
                            
                elif N[n, 0] - M[n, 0] == 0 & N[n, 1] - M[n, 1] < 0:
                                
                    tree_list_presorted[i] = N
           
                    tree_list_presorted[j] = M
            
                    index_sorted[i] = j
                
                    index_sorted[j] = i
                            
                    i = 0
                    
                    j = 1
                            
                    break
                        
                elif N[n, 0] - M[n, 0] == 0 & N[n, 1] - M[n, 1] == 0 & N[n, 2] - M[n, 2] > 0:
                            
                    j = j + 1        
                            
                    break
                                
                elif N[n, 0] - M[n, 0] == 0 & N[n, 1] - M[n, 1] == 0 & N[n, 2] - M[n, 2] < 0:
                                
                    tree_list_presorted[i] = N
           
                    tree_list_presorted[j] = M
            
                    index_sorted[i] = j
                
                    index_sorted[j] = i
                            
                    i = 0
                    
                    j = 1
                            
                    break
                        
                elif N[n, 0] - M[n, 0] == 0 & N[n, 1] - M[n, 1] == 0 & N[n, 2] - M[n, 2] == 0:
                            
                    n = n + 1
        
        else:
            
            j = j + 1
        
        if j == L:
            
            i = i + 1
            
    tree_list_sorted = [tree_list_presorted[k] for k in range(len(index_sorted)) ]
                
    return index_sorted 

#--

def get_total_list_degree_based(tree_list):
    
    inx = get_total_degree_based(tree_list)
    
    aux_list = []

    for i in range(len(inx)):

        j = inx[i]
        
        aux_list.append(tree_list[j])
        
    return aux_list 

#-- 

def get_total_list_evaluation_based(tree_list, a = 2, b = 1):
    
    x, y = symbols('x y')

    poly_unrooted_list = []
    
    sorted_unrooted_list = []

    for i in range(len(tree_list)):
        
        T = tree_list[i]
    
        poly_unrooted_list.append([find_factored_poly_unrooted(T).subs(x, a).subs(y, b), T, i])
                      
    sorted_unrooted_list = sorted(poly_unrooted_list, key = lambda k: k[0])
        
    #return [sorted_unrooted_list[i][1] for i in range(len(sorted_unrooted_list))]

    return sorted_unrooted_list

#--

def smoothing(G):
    
    H = G.copy()
    
    for n in range(H.order()):
        
        if H.degree[n] == 2:
            
            u, v = H.neighbors(n)
            
            H.remove_node(n)
            
            H.add_edge(u, v)
            
    return H

#--

def hamming_iso(T_1, T_2):
    
    if len(T_1) != len(T_2):
    
        return print('The two tree lists have different lengths.')
    
    else:
        
        return sum([not nx.is_isomorphic(T_1[i], T_2[i]) for i in range(len(T_1))]), sum([get_leaf_number(T_1[i]) != get_leaf_number(T_1[i]) for i in range(len(T_1))]), sum([abs(T_1[i].order() - T_2[i].order()) for i in range(len(T_1))]), sum([get_leaf_number(T_1[i]) - get_leaf_number(T_2[i]) for i in range(len(T_1))])
    
#--  

def get_poset_path_to_star(n):

    path_po = nx.Graph()

    tree_list = list(nx.nonisomorphic_trees(n))

    tree_paths_p = []

    tree_path = [[]]

    for tree_index in range(len(tree_list)): 
    
        tree_paths_p.append(len(tree_path))
    
        tree_path = [[tree_list[tree_index], tree_index]]
    
        while len(nx.algorithms.isomorphism.tree_isomorphism(tree_path[-1][0], nx.path_graph(n))) == 0:
    
            tree_path.append([step_to_path(tree_path[-1][0])])
        
            for tree in range(len(tree_list)): 
        
                if len(nx.algorithms.isomorphism.tree_isomorphism(tree_path[-1][0], tree_list[tree])) > 0:
            
                    tree_path[-1].append(tree)
                
                    path_po.add_edge(tree_path[-2][1], tree_path[-1][1])

    tree_paths_p.append(len(tree_path))

    tree_paths_p.pop(0)
    
    return path_po, tree_paths_p
    
#--
    
def get_poset_star_to_path(n):

    star_po = nx.Graph()

    tree_list = list(nx.nonisomorphic_trees(n))

    tree_paths_s = []

    tree_path = [[]]

    for tree_index in range(len(tree_list)): 

        tree_paths_s.append(len(tree_path))
    
        tree_path = [[tree_list[tree_index], tree_index]]
    
        while len(nx.algorithms.isomorphism.tree_isomorphism(tree_path[-1][0], nx.star_graph(n - 1))) == 0:
    
            tree_path.append([step_to_star(tree_path[-1][0])])
        
            for tree in range(len(tree_list)): 
        
                if len(nx.algorithms.isomorphism.tree_isomorphism(tree_path[-1][0], tree_list[tree])) > 0:
            
                    tree_path[-1].append(tree)
                
                    star_po.add_edge(tree_path[-2][1], tree_path[-1][1])

    tree_paths_s.append(len(tree_path))

    tree_paths_s.pop(0)
    
    return star_po, tree_paths_s

#--

def get_spanning_tree_u_w(G):
    
    node_set = set(G.nodes())
    
    x0 = random.choice(tuple(node_set))
    
    x1 = x0
    
    while x1 == x0:
        
        x1 = random.choice(tuple(node_set))
    
    node_set.remove(x1)
    
    tnodes = {x1}
    
    tedges = []
    
    current = x0
    
    current_path = [x0]
    
    current_edges = []
    
    while node_set != set():
        
        next = random.choice(list(G.neighbors(current)))
        
        current_edges.append((current,next))
        
        current = next
        
        current_path.append(next)

        if next in tnodes:
        
            for x in current_path[:-1]:
            
                node_set.remove(x)
                
                tnodes.add(x)
            
            for ed in current_edges:
            
                tedges.append(ed)
            
            current_edges = []
            
            if node_set != set():
            
                current = random.choice(tuple(node_set))
            
            current_path = [current]

        if next in current_path[:-1]:
            
            current_path.pop()
            
            current_edges.pop()
            
            for i in range(len(current_path)):
            
                if current_edges != []:
                    
                    current_edges.pop()
                
                if current_path.pop() == next:
                
                    break
            
            if len(current_path) > 0:
            
                current=current_path[-1]
            
            else:
            
                current = random.choice(tuple(node_set))
                
                current_path = [current]

    return tedges

#--

def get_spanning_tree_u_ab(G):
    
    node_set = set(G.nodes())
    
    x0 = random.choice(tuple(node_set))

    node_set.remove(x0)

    current = x0
    
    tedges = []

    while node_set != set():
        
        next = random.choice(list(G.neighbors(current)))
        
        if next in node_set:
            
            node_set.remove(next)
            
            tedges.append((current, next))
        
        current = next

    return tedges

#--

def tree_cycle_walk_cut(T):

    tempo = 0
    
    tedges = set(T.edges())
    
    while tempo == 0:
    
        edge = (random.choice(tuple(T.nodes())), random.choice(tuple(T.nodes())))
        
        if (edge[0], edge[1]) not in tedges and (edge[1], edge[0]) not in tedges:
        
            tempo = 1
            
            T.add_edge(edge[0], edge[1])
            
            ncycle = nx.find_cycle(T, edge[0])
            
            cutedge = random.choice(tuple(ncycle))
            
            T.remove_edge(cutedge[0], cutedge[1])
    
    return T

#--

def my_incidence_matrix(G, nodelist = None, edgelist = None, oriented = False, weight = None):
    """Returns incidence matrix of G.

    The incidence matrix assigns each row to a node and each column to an edge.
    For a standard incidence matrix a 1 appears wherever a row's node is
    incident on the column's edge.  For an oriented incidence matrix each
    edge is assigned an orientation (arbitrarily for undirected and aligning to
    direction for directed).  A -1 appears for the source (tail) of an edge and
    1 for the destination (head) of the edge.  The elements are zero otherwise.

    Parameters
    ----------
    G : graph
       A NetworkX graph

    nodelist : list, optional   (default= all nodes in G)
       The rows are ordered according to the nodes in nodelist.
       If nodelist is None, then the ordering is produced by G.nodes().

    edgelist : list, optional (default= all edges in G)
       The columns are ordered according to the edges in edgelist.
       If edgelist is None, then the ordering is produced by G.edges().

    oriented: bool, optional (default=False)
       If True, matrix elements are +1 or -1 for the head or tail node
       respectively of each edge.  If False, +1 occurs at both nodes.

    weight : string or None, optional (default=None)
       The edge data key used to provide each value in the matrix.
       If None, then each edge has weight 1.  Edge weights, if used,
       should be positive so that the orientation can provide the sign.

    Returns
    -------
    A : SciPy sparse matrix
      The incidence matrix of G.

    Notes
    -----
    For MultiGraph/MultiDiGraph, the edges in edgelist should be
    (u,v,key) 3-tuples.

    "Networks are the best discrete model for so many problems in
    applied mathematics" [1]_.

    References
    ----------
    .. [1] Gil Strang, Network applications: A = incidence matrix,
       http://videolectures.net/mit18085f07_strang_lec03/
    """
    import scipy as sp
    import scipy.sparse  # call as sp.sparse

    if nodelist is None:
        nodelist = list(G)
    if edgelist is None:
        if G.is_multigraph():
            edgelist = list(G.edges(keys=True))
        else:
            edgelist = list(G.edges())
    A = sp.sparse.lil_matrix((len(nodelist), len(edgelist)))
    node_index = {node: i for i, node in enumerate(nodelist)}
    for ei, e in enumerate(edgelist):
        (u, v) = e[:2]
        if u == v:
            continue  # self loops give zero column
        try:
            ui = node_index[u]
            vi = node_index[v]
        except KeyError as err:
            raise nx.NetworkXError(
                f"node {u} or {v} in edgelist but not in nodelist"
            ) from err
        if weight is None:
            wt = 1
        else:
            if G.is_multigraph():
                ekey = e[2]
                wt = G[u][v][ekey].get(weight, 1)
            else:
                wt = G[u][v].get(weight, 1)
        if oriented:
            A[ui, ei] = -wt
            A[vi, ei] = wt
        else:
            A[ui, ei] = wt
            A[vi, ei] = wt
    import warnings

    warnings.warn(
        "incidence_matrix will return a scipy.sparse array instead of a matrix in Networkx 3.0.",
        FutureWarning,
        stacklevel=2,
    )
    # TODO: Rm sp.sparse.csc_matrix in Networkx 3.0
    return A.asformat("csc")

#--

def simulate_stirling(n, m = 100, seed = 1764):
    
    random.seed(seed)
    
    T = nx.random_tree(n, seed)
    
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
 
        Results.append([n, k, sum(stir) * special.comb(n - 1, k - 1, exact = True)/m])
    
    return Results

#--

def simulate_stirling_diff(n, m = 100, seed = 1764):
    
    random.seed(seed)
    
    T = nx.random_tree(n, seed)
    
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
 
        Results.append([n, k, Stir[-(n - k + 1)] - sum(stir) * special.comb(n - 1, k - 1, exact = True)/m])
    
    return Results