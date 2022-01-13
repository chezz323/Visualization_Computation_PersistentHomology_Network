#import shapefile
import pandas as pd
import plotly.graph_objects as go
import networkx as nx
import numpy as np
from pyproj import Proj, transform

# read data
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
'''
w = []
num_w = 2
for i in range(0, num_w+1):
    w[i] = (max_weight/num_w)*i
'''
#print(Nsize)
#print(nodes.shape)
#print(links.shape)
#print(group)

#print(nodes.head())
#print(links.head())

G = nx.Graph()

for idx, row in nodes.iterrows():
    # add node to graph G
    G.add_node(row['name'], Label=row['name'], group=row['group'], nodesize=row['nodesize'])

thd = 25

for idx, row in links.iterrows():
    # Calculate the distance between Source and Target Nodes
    G.add_edge(row['source'], row['target'], weight=row['value'])
    #weights.append(row['value'])

print(nx.info(G))
for idx, row in nodes.iterrows():
    print("cycle list: ", list(nx.find_cycle(G, row['name'])))

#print(G.nodes)
#print(G.edges)
#print(G.edges[('html', 'css')])

pos = nx.spring_layout(G, k=0.5, iterations=50)

for n, p in pos.items():
    G.nodes[n]['pos'] = p

edge_trace = go.Scatter(
    x=[],
    y=[],
    line=dict(width=0.5, color='#F88'),
    hoverinfo='none',
    mode='lines',
    name='weight: <='+str(thd))

edge_trace1 = go.Scatter(
    x=[],
    y=[],
    line=dict(width=0.5, color='#888'),
    hoverinfo='none',
    mode='lines',
    name='weight: >'+str(thd))

#weights=[]
#nweights=[]

for edge in G.edges():
    x0, y0 = G.nodes[edge[0]]['pos']
    x1, y1 = G.nodes[edge[1]]['pos']
    if G.edges[edge]['weight'] <= thd:
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])
    if G.edges[edge]['weight'] > thd:
        edge_trace1['x'] += tuple([x0, x1, None])
        edge_trace1['y'] += tuple([y0, y1, None])

    #weights.append(G.edges[edge]['weight'])
    #nweights.append(G.edges[edge]['weight']/100)

#print(weights)
#print(nweights)

#edge_trace['line']=dict(width=nweights, color='#888')
#print(edge_trace)

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







#print(nx.shortest_path(G,'java','html'),"length: ", nx.shortest_path_length(G,'java','html'))
#print(nx.eccentricity(G))

fig = go.Figure(data=[edge_trace, edge_trace1, node_trace],
             layout=go.Layout(
                title='<br>Stack Overflow Tag Network',
                titlefont=dict(size=16),
                showlegend=True,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text=" ",
                    showarrow=False,
                    xref="paper", yref="paper") ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

fig.show()



