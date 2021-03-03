

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pylab as plt

def entropy_of_node(df):
    list = df["id"].unique()
    entropy = 0.0
    for value in list:
        num = df["id"].value_counts()[value]
        deno = len(df)
        p = (num/deno)
        entropy = entropy + (-p*(np.log2(p)))
    return entropy

def best_attribute(df):
    npdata = (df.values)
    total = len(npdata)
    ent = entropy_of_node(df)
    max = -1.0
    for i in range(0, len(npdata[0])-1):
        total_entropy =0.0
        l = np.unique(npdata[:,i])
        for j in l:
            x = npdata[npdata[:, i] == j, -1]
            sublen =len(x)
            one = np.count_nonzero(x==1)
            two = sublen-one
            if(abs(one-two)!=sublen):
                en = -((one/sublen)*np.log2(one/sublen))-((two/sublen)*np.log2(two/sublen))
            else :
                en = 0

            total_entropy = total_entropy + (sublen*en)

        total_entropy = total_entropy/total;
        gain = ent-total_entropy
        if(gain > max):
            max = gain
            attr = i

    return attr;  

def entropy_attribute(df, attribute):
    list = df[attribute].unique()
    total_entropy = 0.0
    for value in list:
        df1 = df[df[attribute] == value]
        en = entropy_of_node(df1)
        total_entropy = total_entropy + (len(df1)*en)
    return (total_entropy/len(df))



def generate_tree(df, tree, h):
    attr = df.columns[best_attribute(df)]
    list = df[attr].unique()
    tree = {}
    tree[attr] = {}
    if(h==0):
        if(len(list) > 1):
            grp = np.maximum(df["id"].value_counts()[1],
                                df["id"].value_counts()[2])
            if(grp==df["id"].value_counts()[1]):
                tree[attr] = 1
            else:
                tree[attr] = 2
            return tree
        else:
            tree[attr] = list[0]
            return tree
    for value in list:
        df1 = df[df[attr] == value]
        if(h == 1):
            list = df1["id"].unique()
            if(len(list) > 1):
                grp = np.maximum(df1["id"].value_counts()[1],
                                 df1["id"].value_counts()[2])
                if(grp==df1["id"].value_counts()[1]):
                    tree[attr][value] = 1
                else:
                    tree[attr][value] = 2
            else:
                tree[attr][value] = list[0]
        else:
            if(entropy_of_node(df1) == 0):
                grp = df1.iloc[0, -1]
                tree[attr][value] = grp
            else:
                tree[attr][value] = generate_tree(df1, tree, h-1)
    return tree

def print_tree(tree,tab=0):
    for attr in tree.keys():
        key = [i for i in tree[attr].keys()]
        
        for att in range(0,len(key)):
            
            for i in range(0,tab):
                print("\t" ,end="")
            if (tab>0):
                print("|",end="")
            print(attr,"= ",end="")
            print(key[att],end="")
            if type(tree[attr][key[att]]) is not dict:
                print(":",end="")
                print(tree[attr][key[att]])
            else:
                print(end="\n")
                print_tree(tree[attr][key[att]],tab+1)

def prediction(tree,test):
    #keys = [i for i in tree.keys()]
     
    for attr in tree.keys():        
        value = test[attr]
        predict = 0
        if type(tree[attr][value]) is not dict:
            predict = tree[attr][value]
            break;
        else:
            predict = prediction(tree[attr][value],test)
    return predict

#taking data in dataframes
traindata_df = pd.read_csv('traindata.txt', sep="\t", header=None)
trainlabel_df = pd.read_csv('trainlabel.txt', header=None)
words_df = pd.read_csv('words.txt', header=None)
testdata_df = pd.read_csv('testdata.txt', sep="\t", header=None)
testlabel_df = pd.read_csv('testlabel.txt', header=None)

#converting into 2 d matrix
index_list = list(traindata_df[0])
word_no = list(traindata_df[1])
label_list = list(trainlabel_df[0])
words = list(words_df[0])
data_list = []
doc_list = [0]*(len(words)+1)
doc_id = 1
for i in range(len(index_list)):
    while(index_list[i] != doc_id):
        data_list.append(doc_list)
        doc_id = doc_id + 1
        doc_list = [0]*(len(words)+1)
    doc_list[word_no[i]-1] = 1
    if(doc_list[-1] == 0):
        doc_list[-1] = label_list[doc_id-1]

data_list.append(doc_list)
data_list[1016][3566] = 2
data_df = pd.DataFrame(data_list)
#adding id as class name since its not in word list
words.append('id')
data_df1=data_df
data_df.columns = words
#for test data
index_list = list(testdata_df[0])
word_no = list(testdata_df[1])
label_list = list(testlabel_df[0])
words = list(words_df[0])
test_list =[]
doc_list = [0]*(len(words)+1)
doc_id = 1
for i in range(len(index_list)):
    while(index_list[i] != doc_id):
        test_list.append(doc_list)
        doc_id = doc_id + 1
        doc_list = [0]*(len(words)+1)
    doc_list[word_no[i]-1] = 1
    if(doc_list[-1] == 0):
        doc_list[-1] = label_list[doc_id-1]

test_list.append(doc_list)
test_df = pd.DataFrame(test_list)
words.append('id')
test_df1=test_df
test_df.columns = words



#training and testing using scikit learn
#using information gain
dtree = DecisionTreeClassifier(criterion="entropy",random_state=1)
dtree.fit(data_df.iloc[:,:-1],data_df.iloc[:]["id"])
accuracy_entropy= dtree.score(test_df.iloc[:,:-1],test_df.iloc[:]["id"])
percent_accuracy =100*accuracy_entropy
print("\nAccuracy using Info gain(scikit) is:- %.4f" %percent_accuracy,"%\n")


h= int(input("Enter height :="))
tree ={}
tree = generate_tree(data_df, tree, h)
print("\nClass 1 = alt.atheism")
print("Class 2 = comp.graphics\n")
print_tree(tree)

print("\nGenerating graph (will take 3 to 4 mins)")
#graph and predictions
label_list = list(trainlabel_df[0])
test_label_list = list(testlabel_df[0])
test_accuracy =[]
train_accuracy =[]
maximum_depth = 0
i=0
while(1):
    i=i+1
    tree ={}
    tree = generate_tree(data_df, tree, i)
    test_prediction =[]
    train_prediction =[]
    for j in range(len(test_df)):
        test_prediction.append(prediction(tree,test_df.iloc[j,:]))
    for j in range(len(data_df)):
        train_prediction.append(prediction(tree,data_df.iloc[j,:]))
    m =0
    for k in range(len(label_list)):
        if(label_list[k]==train_prediction[k]):
            m=m+1
    acc = m/len(label_list)
    train_accuracy.append(acc)
    m=0
    for k in range(len(test_label_list)):
        if(test_label_list[k]==train_prediction[k]):
            m=m+1
    test_accuracy.append(m/len(test_label_list))
    if(acc==1):
       maximum_depth = i
       break;


print("Maximum depth =",maximum_depth)
#plotting the data
plt.plot(list(range(20)),train_accuracy,marker="o",label="Train accuracy")
plt.plot(list(range(20)),test_accuracy,marker="x",label="Test accuracy")
plt.ylim(0,1.5)
plt.legend()
plt.show()


