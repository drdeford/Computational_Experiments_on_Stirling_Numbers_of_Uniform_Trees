# Stirling Numbers of Uniform Trees and Related Computational Experiments
(under development)

Python code for sampling and evaluating cycle covers of trees.
  
Contents:
- **FKT.py**: An implementation of the FKT algorithm for constructing a pfaffian orientation of a planar graph.
- **uniforrm\_matching.py**: An implementation of the uniform sampling method from a paper of Jerrum, Valiant, and Vazirani.
- **uniform\_cycle\_cover.py**: An implementation of a uniform sampling method for cycle covers of planar bipartite graphs.
- **uniform\_matching\_symbolic.py**: A method for enumerating the k-Stirling numbers of trees using a permanent-determinant approach.
- **spanning\_tree\_metrics.py**: An implementation of an MCMC version of the cycle basis walk on spanning trees with Metropolis-Hastings for interpolating between stars and paths.
- **Classification--{Betweenness, Closeness, Stirling, All}--{Full_Set, Sampling}.ipynb**: These Jupyter notebooks use statistical learning methods to classify trees (into 2 classes: path-like or star-like) with global betweenness centrality (**Betweenness**), global closeness centrality (**Closeness**), the Stirling numbers of the first kind (**Stirling**), or all three (**All**) as predictors. Two different data sets are used in these notebooks: The first data set (**Full_Set**) consists of all non-isomorphic trees of order 15 and the second data set consists of 500 non-isomorphic trees of order 18.
- **Regression--{All, Subset}--{Full_Set, Sampling}.ipynb**: These Jupyter notebooks use statistical learning methods to predict the Stirling numbers of the first kind using $\log10(P (T ; 2, 1))$ (base 10 logarithm of the distinguishing polynomial of $T$ evaluated at $x = 2$ and $y =1$), global closeness centrality, global betweenness centrality, and class as predictors.
- **tree\_functions\_2.py**: All the functions defined by the authors and used in the above notebooks are in this file.

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
  
