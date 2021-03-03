
import numpy as np
import pandas as pd
import pickle


def get_cluster_entropy(model):
    cluster_ratio = []
    cluster_entropy = 0.0
    for i in range(len(model)):
        l = len(model[i])
        cluster_ratio.append(l/150)
    for i in cluster_ratio:
        cluster_entropy += -1*i*np.log2(i)
    return cluster_entropy, cluster_ratio


def get_IF(model, data, en):
    total_mut = 0.0
    for i in range(len(model)):
        mut_info = 0.0
        p = len(model[i])/150
        df = data[model[i]]
        for j in df.unique():
            x = df.value_counts()[j]
            mut_info += -1*(x/len(model[i]))*np.log2(x/len(model[i]))
        mut_info = p*mut_info
        total_mut += mut_info

    return en-total_mut

def print_model(model,data):
    for i in range(len(model)):
        df = data[model[i]]
        x = df.value_counts().idxmax()
        print("Cluster ",i+1,":","(Majority class :",x,")","length :",len(model[i]),"\n",model[i])

data = pd.read_csv("AAAI.csv")
columns = list(data.columns)
classlist = list(data["High-Level Keyword(s)"].unique())
class_data = data.loc[:, "High-Level Keyword(s)"]
class_data.columns = ["cls"]
# class_data = class_data.replace(classlist, [0, 1, 2, 3, 4, 5, 6, 7, 8])
testfile = open('Ass3q1', 'rb')
model = pickle.load(testfile)
testfile.close()
model1 = model[0]
model2 = model[1]
testfile1 = open('Ass3q2', 'rb')
model3 = pickle.load(testfile1)
testfile1.close()

class_ratio = []
for i in class_data.unique():
    # print(i, "=>", class_data.value_counts()[i])
    class_ratio.append(class_data.value_counts()[i]/150)

class_entropy = 0.0
for i in class_ratio:
    class_entropy += -1*i*np.log2(i)
cluster_entropy, cluster_ratio = get_cluster_entropy(model1)
mutual_info = get_IF(model1, class_data, class_entropy)
nmi1 = (2*mutual_info)/(class_entropy+cluster_entropy)
print("NMI for single linkage clustering is: ", nmi1)
cluster_entropy, cluster_ratio = get_cluster_entropy(model2)
mutual_info = get_IF(model2, class_data, class_entropy)
nmi2 = (2*mutual_info)/(class_entropy+cluster_entropy)
print("NMI for complete linkage clustering is: ", nmi2)
cluster_entropy, cluster_ratio = get_cluster_entropy(model3)
mutual_info = get_IF(model3, class_data, class_entropy)
nmi3 = (2*mutual_info)/(class_entropy+cluster_entropy)
print("NMI for Girvan-Newman clustering is: ", nmi3)

