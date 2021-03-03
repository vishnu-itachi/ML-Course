
import numpy as np
import pandas as pd
import networkx as nx
import pickle

#calculating similarity matrix for pairs of doc

def sim_matrix(df):
    l = len(df)
    mat = np.zeros(l*l).reshape(l, l)
    for i in range(l):
        for j in range(l):
            if(i != j):
                num = len(list(set(df["Topics"][i]) & set(df["Topics"][j])))
                deno = len(list(set(df["Topics"][i]) | set(df["Topics"][j])))
                mat[i][j] = num/deno
    return mat

#creating graph based on threshold
def create_graph(mat, threshold):
    g = nx.Graph()
    for i in range(len(mat)):
        for j in range(i+1, len(mat)):
            g.add_nodes_from([i, j])
            if(mat[i][j] >= threshold):
                g.add_edge(i, j)
    return g

def print_model(model,data):
    for i in range(len(model)):
        df = data[model[i]]
        x = df.value_counts().idxmax()
        print("Cluster ",i+1,":","(Majority class :",x,")","length :",len(model[i]),"\n",model[i])

#reading data and converting to dataframe
data = pd.read_csv("AAAI.csv")
for i in range(150):
    data.iloc[i, 2] = data.iloc[i, 2].split('\n')
train_data = data.loc[:, ["Topics"]]
mat = sim_matrix(train_data)

#for getting majority class of cluster
class_data = data.loc[:, "High-Level Keyword(s)"]
class_data.columns = ["cls"]

g = nx.Graph()
threshold = 0.20
g = create_graph(mat, threshold)
# print(nx.info(g))
#creating model
while(1):
    l = len(list(nx.connected_component_subgraphs(g)))
    if(l == 9):
        break
    ed = nx.edge_betweenness_centrality(g)
    rm_edge = list(sorted(ed, key=ed.get, reverse=True)[0])
    g.remove_edge(rm_edge[0], rm_edge[1])
model = []
for subgraph in list(nx.connected_component_subgraphs(g)):
    model.append(list(subgraph.nodes()))
    #print(subgraph.nodes())

#storing in file for part3
testfile = open('Ass3q2', 'wb')
pickle.dump(model, testfile)
testfile.close()

#printing model
print("model3 Girvan-Newman clustering with threshold :",threshold,"\n")
print_model(model,class_data)
