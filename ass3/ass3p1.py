
import numpy as np
import pandas as pd
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

# complete linkage
def min_simlarity(l, m, mat):
    sim = 2
    for i in range(len(l)):
        for j in range(len(m)):
            if (sim >= mat[l[i]][m[j]]):
                sim = mat[l[i]][m[j]]
    return sim

# single linkage
def max_simlarity(l, m, mat):
    sim = -1
    for i in range(len(l)):
        for j in range(len(m)):
            if (sim <= mat[l[i]][m[j]]):
                sim = mat[l[i]][m[j]]
    return sim

#gives the index of cluster which is to be joined
def join_cluster_min(model, mat):
    lm = len(model)
    sim = -1
    for i in range(lm):
        for j in range(i+1, lm):
            simc = min_simlarity(model[i], model[j], mat)
            if (i != j and simc > sim):
                sim = simc
                c1 = i
                c2 = j
    return c1, c2

#same as above one for single one for complete
def join_cluster_max(model, mat):
    lm = len(model)
    sim = -1
    for i in range(lm):
        for j in range(i+1, lm):
            simc = max_simlarity(model[i], model[j], mat)
            if (i != j and simc > sim):
                sim = simc
                c1 = i
                c2 = j
    return c1, c2

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

#for getting majority class of cluster
class_data = data.loc[:, "High-Level Keyword(s)"]
class_data.columns = ["cls"]

mat = sim_matrix(train_data)

model1 = []
model2 = []
for i in range(150):
    l = [i]
    model1.append(l)
    model2.append(l)

#creating model1 single linkage
while(1):
    if(len(model1) == 9):
        break
    c1, c2 = join_cluster_max(model1, mat)
    model1[c1] = list(set(model1[c1]) | set(model1[c2]))
    model1.remove(model1[c2])

#creating model1 complete linkage
while(1):
    if(len(model2) == 9):
        break
    c1, c2 = join_cluster_min(model2, mat)
    model2[c1] = list(set(model2[c1]) | set(model2[c2]))
    model2.remove(model2[c2])

#storing in file for part3
model = []
model.append(model1)
model.append(model2)
testfile = open('Ass3q1', 'wb')
pickle.dump(model, testfile)
testfile.close()

#printing models
print("model1 single linkage:\n")
print_model(model1,class_data)
print("\nmodel2 complete linkage:\n")
print_model(model2,class_data)


