#import package
import pandas as pd
import plotly.graph_objects as go
import networkx as nx
from networkx.utils import *
import gudhi as gd
from itertools import combinations
import matplotlib.pyplot as plot
import numpy as np

# read the data set
# 1. node data
# 2. link information
nodes = pd.read_csv('stack_network_nodes.csv')
links = pd.read_csv('stack_network_links.csv')

# node size normalization [min, max] to [Nmin, Nmax]
Nsize = []
group = []
dic_group = {1: 'aqua', 2: 'chocolate', 3:'tomato', 4:'coral',
             5: 'darkcyan', 6: 'gold', 7: 'lightblue', 8: 'grey',
             9: 'ivory', 10: 'crimson', 11: 'green', 12: 'darkred',
             13: 'lime', 14: 'silver', 15: 'plum', 16: 'violet'}

min = nodes.nodesize[0]
max = nodes.nodesize[0]
Nmin = 10
Nmax = 50
max_weight = links.value[0]

for i in range(0, nodes.shape[0]):
    if min >= nodes.nodesize[i]: min = nodes.nodesize[i]
    if max <= nodes.nodesize[i]: max = nodes.nodesize[i]

for i in range(0, nodes.shape[0]):
    Nsize.append(Nmin+(Nmax-Nmin)*(nodes.nodesize[i]-min)/(max-min))
    group.append(dic_group[nodes.group[i]])

for i in range(0, links.shape[0]):
    if max_weight <= links.value[i]: max_weight = links.value[i]

# declare graph using data
G = nx.Graph()

# add nodes to graph G
for idx, row in nodes.iterrows():
    G.add_node(row['name'], Label=row['name'], group=row['group'], nodesize=row['nodesize'])

# add edges to graph G
for idx, row in links.iterrows():
    G.add_edge(row['source'], row['target'], weight=row['value'])


##Network Visualization
pos = nx.spring_layout(G, k=0.5, iterations=50)

for n, p in pos.items():
    G.nodes[n]['pos'] = p

edge_trace = go.Scatter(
    x=[],
    y=[],
    line=dict(width=0.5, color='#888'),
    hoverinfo='none',
    mode='lines')

for edge in G.edges():
    x0, y0 = G.nodes[edge[0]]['pos']
    x1, y1 = G.nodes[edge[1]]['pos']
    edge_trace['x'] += tuple([x0, x1, None])
    edge_trace['y'] += tuple([y0, y1, None])

node_trace = go.Scatter(
    x=[],
    y=[],
    text=[],
    mode='markers',
    hoverinfo='text',
    #legend=True,
    marker=dict(
        showscale=False,
        colorscale='RdBu',
        #colorbar=None,
        reversescale=True,
        color=group,
        size=Nsize,
        line=dict(width=2)))

for node in G.nodes():
    x, y = G.nodes[node]['pos']
    node_trace['x'] += tuple([x])
    node_trace['y'] += tuple([y])

for node, adjacencies in enumerate(G.adjacency()):
    node_trace['marker']['color']+=tuple([len(adjacencies[1])])
    node_info = adjacencies[0] +' # of connections: '+str(len(adjacencies[1])) +\
                ' | Node size: ' + str(nodes.nodesize[node]) +\
                ' | Group: ' + str(nodes.group[node])
    node_trace['text']+=tuple([node_info])


# plot the network data using package "plotly"
fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                title='<br>Stack Overflow Tag Network',
                titlefont=dict(size=16),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text=" ",
                    showarrow=False,
                    xref="paper", yref="paper") ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

fig.show()
# end of network visualization

##Compute Persistent homology and Betti Numbers

# declare function for finding all cycle in graph G
def cycle_basis(G, root=None):
    gnodes=set(G.nodes())
    cycles=[]
    while gnodes:  # loop over connected components
        if root is None:
            root=gnodes.pop()
        stack=[root]
        pred={root:root}
        used={root:set()}
        while stack:  # walk the spanning tree finding cycles
            z=stack.pop()  # use last-in so cycles easier to find
            zused=used[z]
            for nbr in G[z]:
                if nbr not in used:   # new node
                    pred[nbr]=z
                    stack.append(nbr)
                    used[nbr]=set([z])
                elif nbr == z:        # self loops
                    cycles.append([z])
                elif nbr not in zused:# found a cycle
                    pn=used[nbr]
                    cycle=[nbr,z]
                    p=pred[z]
                    while p not in pn:
                        cycle.append(p)
                        p=pred[p]
                    cycle.append(p)
                    cycles.append(cycle)
                    used[nbr].add(z)
        gnodes-=set(pred)
        root=None
    return cycles

# find all cycle
cycles = cycle_basis(G)

# st: simplex tree by package gudhi
# st only get input as integer, it is required to transform nodes to integer
st = gd.SimplexTree()
dic = {}
for idx, row in nodes.iterrows():
    # change index string to integer
    dic[row['name']] = idx
    # insert all nodes into st
    # since we assume that every node exists from the beginning, filtration should be 0
    st.insert([dic[row['name']]], filtration=0)


# insert all nodes into st, which filtration value is weight of each edge
for idx, row in links.iterrows():
    st.insert([dic[row['source']], dic[row['target']]], filtration=row['value'])

# cycles what we found above are set of strings so it is required to transfomr into integers
# weight of cycle is the largest weight of edge in cylce
cyclesToNum = []
for cycle in cycles:
    cycle_weight = 0.0
    cycleToNum = []
    for edge in list(combinations(cycle, 2)):
        try:
            if cycle_weight <= G[str(edge[0])][str(edge[1])]['weight']: cycle_weight = G[str(edge[0])][str(edge[1])]['weight']
        except:
            print('There is no edge!', edge[0], edge[1])
    for node in cycle:
        cycleToNum.append(dic[node])
    #print(cycleToNum, cycle_weight)
    #print(cycle_weight)
    if len(cycle)==3:st.insert(cycleToNum, filtration=cycle_weight)
    cyclesToNum.append(cycleToNum)


# initialize all filtration based on G
st.initialize_filtration()
st_list = st.get_filtration()

# print information of simplex tree of G
print("dimension=", st.dimension())
print("num_simplices=", st.num_simplices())
print("num_vertices=", st.num_vertices())
print("skeleton[2]=", st.get_skeleton(2))
print("skeleton[1]=", st.get_skeleton(1))
print("skeleton[0]=", st.get_skeleton(0))

# based on simplex tree, compute persistence homology
# it makes all persistences of form 'dimension, (birt, death)'
ph_data = st.persistence()
st.write_persistence_diagram('test.txt')

# show persistence barcode
gd.plot_persistence_barcode(ph_data, legend=True)
plot.show()

# show persistence diagram
gd.plot_persistence_diagram(ph_data, band=0.2, legend=True)
plot.show()

# show both persistence barcode and diagram
fig, axes = plot.subplots(nrows=1, ncols=2)
gd.plot_persistence_barcode(ph_data, axes = axes[0], legend=True)
gd.plot_persistence_diagram(ph_data, axes = axes[1], legend=True)
fig.suptitle("barcode versus diagram")
plot.show()

# compute betti numbers

# print betti numbers of end state
print(st.betti_numbers())
print(st.persistent_betti_numbers(23.7,23.7))
