from FKT import *

import networkx as nx #Requires at least networkx 2.3+

import matplotlib.pyplot as plt

import random

import math

import numpy as np

import time

from sympy import *

import sympy as sp

#--

def select_edge_clean(H, slow = True):

    G2 = H.copy()
    
    ccs = list((G2.subgraph(c) for c in nx.connected_components(G2)))
    
    edges = []
    
    adds2 = [[None, 0]]
    
    rsum = 0    
    
    for cc in ccs:
        
        G = cc.copy()

        if len(list(G.nodes())) == 1:
           
            break
        
        match_all = FKT((nx.adjacency_matrix(G)).todense())
        
        if match_all == 0:
            
            break
    
        elist = list(G.edges())

        temp = 0

        for edge in elist:

            G.remove_nodes_from([edge[0], edge[1]])

            C = [G.subgraph(c) for c in nx.connected_components(G)]

            compprod = 1
            
            for comp in C:
                
                if len(list(comp.nodes())) % 2 == 1:
                    
                    compprod = 0

                else:
                    
                    compprod = compprod * round(FKT((nx.adjacency_matrix(comp)).todense()))
                    
            if not C:
                
                rsum += match_all
                
                adds2.append([edge, rsum])
                
            elif compprod > 0:
                
                rsum += compprod
                
                adds2.append([edge, rsum])          
                
            G = cc.copy()

    r = random.randint(1, rsum)

    for x in range(len(adds2) - 1):
        
        if int(adds2[x + 1][1]) >= r and int(adds2[x][1]) < r:

            return adds2[x + 1][0]
                          
#--
        
def select_edge(H, slow=True):
    
    #print("new call")
    
    G2 = H.copy()
    
    ccs = list((G2.subgraph(c) for c in nx.connected_components(G2)))
    
    #print(len(ccs))

    edges = []
    
    adds2 = [[None, 0]]
    
    rsum = 0    
    
    for cc in ccs:
        
        G = cc.copy()
        
        if len(list(G.nodes())) == 1:
           
            break
        
        match_all = FKT((nx.adjacency_matrix(G)).todense())
        
        if match_all == 0:
            
            break
        
        #print("did all")

        elist = list(G.edges())
        
        #print(len(elist))
        
        temp = 0

        for edge in elist:
            
            #G.remove_edge(edge[0],edge[1])
            
            G.remove_nodes_from([edge[0], edge[1]])
            
            #print("edge", edge)
            
            C = [G.subgraph(c) for c in nx.connected_components(G)]

            compprod = 1
            
            for comp in C:
                
                if len(list(comp.nodes())) % 2 == 1:
                    
                    compprod = 0
                    
                    #break
                
                else:
                    
                    compprod = compprod * round(FKT((nx.adjacency_matrix(comp)).todense()))
                    
                    #FKT((nx.adjacency_matrix(comp)).todense())
                    
                   
            #if compprod == match_all:
            
            #    edges.append(edge)
            
            #    rsum += compprod
                
            #    adds2.append([edge, rsum])
                
            #else:
            
            #   rsum += compprod
                
            #    adds2.append([edge, rsum])
            
            if not C:
                rsum+= match_all
                adds2.append([edge,rsum])
                
            elif compprod > 0:
                
                rsum += compprod
                
                adds2.append([edge, rsum])
            
            #print(edge,compprod)
    
                #for i in range(compprod):
                
                #    adds.append(edge)
            
            #print(adds)
            
            #probs.append(probs[-1] + compprod/match_all)
            
            #probs.append(probs[-1] + compprod)
            
            #G.add_edge(edge[0], edge[1])
            
            G = cc.copy()
    
    #print(adds2)            
    
    r = random.randint(1, rsum)

    #print("r", r)
    
    for x in range(len(adds2) - 1):
        
        if int(adds2[x + 1][1]) >= r and int(adds2[x][1]) < r:
            
            #print(adds2[x + 1])
            
            return adds2[x + 1][0]
            
            #edges.append(adds2[x + 1][0])
            
            #print(edge)
            
            #temp += 1
            
            #print(temp)
        
        #avlsdkn   
        
        #if adds:
            
            #toremove = random.choice(adds)
            
            #print(toremove)
        
        #edges.append(toremove)
        
        #edges.append(random.choice(adds))
    
    #print(edges)
        
    #return edges

"""        
    probs.pop(0)
    
    print(probs)
    
    #r = random.random()
    
    r = random.randint(0, rsum)
    
    print(r)
    
    for x in range(len(elist)):
        
        if probs[x]>r:
            
            return elist[x]
    
    return random.choice(elist)
"""       

#--

def select_edge_leaves(H, slow=True):
    
    #print("new call")
    
    G2 = H.copy()
    
    ccs = list((G2.subgraph(c) for c in nx.connected_components(G2)))
    
    #print(len(ccs))

    edges = []
    
    adds2 = [[None, 0]]
    
    rsum = 0    
    
    for cc in ccs:

        G=cc.copy()

        if len(list(G.nodes())) == 1:
           
            break
        
        #match_all = FKT((nx.adjacency_matrix(G)).todense())
        
        #if match_all == 0:
        
        #    break
        
        #print("did all")

        elist = list(G.edges())
        
        #print(len(elist))
        
        temp = 0

        for edge in elist:
            
            #G.remove_edge(edge[0], edge[1])
            
            G.remove_nodes_from([edge[0], edge[1]])
            
            #print("edge", edge)
            
            C = [G.subgraph(c) for c in nx.connected_components(G)]

            compprod = 1
            
            for comp in C:
                
                if len(list(comp.nodes())) % 2 == 1:
                    
                    compprod = 0
                    
                    break
                
                else:
                    
                    compprod = compprod * FKT((nx.adjacency_matrix(comp)).todense())
                    
                   
            #if compprod == match_all:
            
            #    #edges.append(edge)
            
            #    rsum += compprod
                
            #    adds2.append([edge, rsum])
                
            #else:
            
            #   rsum += compprod
                
            #    adds2.append([edge, rsum])
            
            if not C:
                
                #rsum += match_all
                
                #adds2.append([edge, rsum])
                
                return(edge)
                
            #elif compprod == match_all:
            
            #    return(edge)
                
            elif compprod > 0:
                
                rsum += compprod
                
                adds2.append([edge, rsum])
            
            #print(edge, compprod)
    
                #for i in range(compprod):
                
                #    adds.append(edge)

            #print(adds)
            
            #probs.append(probs[-1] + compprod/match_all)
            
            #probs.append(probs[-1] + compprod)
            
            #G.add_edge(edge[0], edge[1])
            
            G = cc.copy()
    
    #print(adds2)            
    
    r = random.randint(1, math.ceil(rsum))
    
    #r = random.random()*rsum + 1

    #print("r", r)
    
    for x in range(len(adds2) - 1):
        
        if adds2[x + 1][1] >= r and adds2[x][1] < r:
            
            #print(adds2[x + 1])
            
            return adds2[x + 1][0]

#--
        
def uniform_matching(H):
    
    g = H.copy()
    
    nlist = list(g.nodes())
    
    mlist = []
    
    if len(nlist) % 2 == 1:
        
        return []
    
    while len(nlist) > 0:
                
        edge = select_edge(g)
        
        #for edge in edges:
        
        mlist.append(edge)
    
        #if edge[0] in nlist:
        
        nlist.remove(edge[0])
        
        g.remove_node(edge[0])
        
        #if edge[1] in nlist:
        
        nlist.remove(edge[1])

        g.remove_node(edge[1])

        #plt.figure()
        
        #nx.draw(H, pos = {x:x for x in H.nodes()}, node_color = ['r' for x in H.nodes()])
        
        #nx.draw(g, pos = {x:x for x in g.nodes()})
        
        #plt.show()

    return mlist

#--
    
def uniform_cycle_cover(H, two_cycles = False):
    
    g = H.copy()
    
    m1 = uniform_matching(g)
    
    D = nx.DiGraph()
    
    for x in m1:
        
        if (x[0][0] + x[0][1]) % 2 == 0:
            
            D.add_edge(x[0], x[1])
        
        else:
            
            D.add_edge(x[1], x[0])
            
    g = H.copy()
            
    if two_cycles == False:
            
        g.remove_edges_from(m1)
    
    m2 = uniform_matching(g)
    
    for x in m2:
        
        if (x[0][0] + x[0][1]) % 2 == 1:
            
            D.add_edge(x[0], x[1])
        
        else:
            
            D.add_edge(x[1], x[0])

    return D

#--

plt.figure()

dg = uniform_cycle_cover(nx.grid_graph([8, 8]))

nx.draw(dg,pos = {x:x for x in dg.nodes()})

plt.show()

adkvdkjv 

#--

def get_spanning_tree_u_ab(G):
    
    node_set=set(G.nodes())
    
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

t= get_spanning_tree_u_ab(nx.grid_graph([4, 5]))

tgraph = nx.Graph()

tgraph.add_edges_from(t)

plt.figure()

nx.draw(tgraph)

plt.show()

A = nx.adjacency_matrix(tgraph).todense()

st = time.time()

BG = nx.Graph()

for i in range(len(A)):
    
    for j in range(len(A)):
        
        if A[i,j] == 1:
            
            BG.add_edge(i + 1, -(j + 1))
        
        if i == j:
            
            BG.add_edge(i + 1, -(j + 1))
            
nx.draw(BG)

#print(FKT(nx.adjacency_matrix(BG).todense()))

signedmat = FKT_mat(nx.adjacency_matrix(BG).todense())

weightmat = sp.Matrix(signedmat[:])

nlist = list(BG.nodes())

q = sp.Symbol("q")

for i in range(len(nlist)):
    
    for j in range(len(nlist)):
        
        if nlist[i] != -nlist[j]:
            
            weightmat[i, j] = weightmat[i, j]*q #7
           
print((weightmat.det()).factor())

print(time.time() - st)

print(sp.factor_list(weightmat.det()))

print(np.prod([x[0]**int(x[1]/2) for x in sp.factor_list(weightmat.det())[1]]).expand())

#print(math.sqrt(weightmat.det()))
            
#np.savetxt("./outmat4.csv", weightmat,fmt='%d', delimiter = ',', newline='],\n[')

#np.savetxt("./TESTMAT.csv", signedmat,fmt='%d', delimiter=',', newline='],\n[')
