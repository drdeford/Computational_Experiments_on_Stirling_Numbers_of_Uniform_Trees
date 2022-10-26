# Stirling Numbers of Uniform Trees and Related Computational Experiments
(under development)

## Code

Python code for sampling and evaluating cycle covers of trees.

### **FKT.py**

An implementation of the FKT algorithm for constructing a pfaffian orientation of a planar graph.

### **uniforrm\_matching.py**

An implementation of the uniform sampling method from the paper ["Random generation of combinatorial structures from a uniform distribution"](https://www.sciencedirect.com/science/article/pii/030439758690174X), by Jerrum, Valiant, and Vazirani.

### **uniform\_cycle\_cover.py**

An implementation of a uniform sampling method for cycle covers of planar bipartite graphs.

### **uniform\_matching\_symbolic.py**

A method for enumerating the $k$-th Stirling numbers of the first kind for trees using a permanent-determinant approach.

### **spanning\_tree\_metrics.py**

An implementation of an MCMC version of the cycle basis walk on spanning trees with Metropolis-Hastings for interpolating between stars and paths.

### **Probabistic\_Approach.ipynb**

In this notebook, a random tree $T$ is generated. For this tree, $m$ trial are ran. At each trial a $\\{0,1\\}$-column vectors with random entries is generated and a trial is considered a "success" if the condition that it contains exactly $k-1$ many $1$'s is met. The code for generating result for $n \in \\{7, 8, \ldots, 15\\}$ and $m \in \\{10000, 20000\\}$ are included. Moreover, the code produces average running times for $n \in \\{7, \ldots, 15, 24, 25, 26\\}$ and $m \in \\{10000, 20000\\}$. 

### **probabilistic\_approach--average\_running\_times.py** 

This Python script is used to compute the average running times for $n \in \\{5, 6, \ldots, 30\\}$ and $m \in \\{10000, 20000\\}$ for the code in the 
**Probabistic\_Approach.ipynb** notebook. Running time for this Python script is long.

### **Uniform\_Trees--Autocorrelation.ipynb**

This notebook generates the autocorrelation plots for uniform sampling from the tree space based on degree, maximum degreem, betweenness, and closeness with $100000$ iterations, lags up to $1000000$, $p = 0.01$, $q \in \{0.2, 0.4, 0.6, 0.8\}$, and $r \in \{0.2, 0.4, 0.6, 0.8\}$, where each point is displayed in increments of $50$. The blue ribbons in these plots are made of the $95\%$ confidence intervals where the standard deviation is computed according to Bartlett’s formula ([statsmodels.graphics.tsaplots.plot\_acf](https://www.statsmodels.org/dev/generated/statsmodels.graphics.tsaplots.plot_acf.html)).

### **uniform_trees--autocorrelation--longer_runs.py**

This Python script is used to generate the autocorrelation plots for $p = 0.1$, $q, r \in \\{0.2, 0.4, 0.6, 0.8\\}$, and $10000000$ iterations for the code in the **Uniform\_Trees--Autocorrelation.ipynb** notebook. Running time for this Python script is very long. 

### **Classification--{Betweenness, Closeness, Stirling, All}--{Full_Set, Sampling}.ipynb**

These Jupyter notebooks use statistical learning methods to classify trees (into two classes: path-like or star-like) with global betweenness centrality (**Betweenness**), global closeness centrality (**Closeness**), the Stirling numbers of the first kind (**Stirling**), or all three (**All**) as predictors. Two different data sets are used in these notebooks: The first data set (**Full_Set**) consists of all non-isomorphic trees of order 15 and the second data set consists of 500 non-isomorphic trees of order 18.

### **Regression--{All, Subset}--{Full_Set, Sampling}.ipynb**

These Jupyter notebooks use statistical learning methods to predict the Stirling numbers of the first kind using $\log10(P (T ; 2, 1))$ (base 10 logarithm of the distinguishing polynomial of $T$ evaluated at $x = 2$ and $y =1$), global closeness centrality, global betweenness centrality, and class as predictors.

### **tree\_functions\_2.py**

All the functions defined by the authors and used in the above notebooks are in this file.

## Autocorrelation Plots for Uniform Sampling from the Tree Space

### Global Betweenness Centrality

<center> 
<table>
    <tr>
        <td> <img src='https://github.com/drdeford/Stirling_Trees/blob/master/BTW_0.gif' width = '200'></td>
        <td> <img src='https://github.com/drdeford/Stirling_Trees/blob/master/BTW_1.gif' width = '200'> </td>
        <td> <img src='https://github.com/drdeford/Stirling_Trees/blob/master/BTW_2.gif' width = '200'> </td>
    </tr>
    <tr>
        <td> <img src='https://github.com/drdeford/Stirling_Trees/blob/master/BTW_3.gif' width = '200'> </td>
        <td> <img src='https://github.com/drdeford/Stirling_Trees/blob/master/BTW_4.gif' width = '200'> </td>
    </tr>    
</table>
</center>

### Global Closeness Centrality

<center> 
<table>
    <tr>
        <td> <img src='https://github.com/drdeford/Stirling_Trees/blob/master/CLS_0.gif' width = '200'></td>
        <td> <img src='https://github.com/drdeford/Stirling_Trees/blob/master/CLS_1.gif' width = '200'> </td>
        <td> <img src='https://github.com/drdeford/Stirling_Trees/blob/master/CLS_2.gif' width = '200'> </td>
    </tr>
    <tr>
        <td> <img src='https://github.com/drdeford/Stirling_Trees/blob/master/CLS_3.gif' width = '200'> </td>
        <td> <img src='https://github.com/drdeford/Stirling_Trees/blob/master/CLS_4.gif' width = '200'> </td>
    </tr>
</table>
</center>
  
