

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

def entropy_of_node(df):
    list=df["profitable"].unique()
    entropy = 0.0
    for value in list:
        num=df["profitable"].value_counts()[value]
        deno = len(df)
        p=(num/deno)
        entropy = entropy + (-p*(np.log2(p)))
    return entropy
   
def entropy_attribute(df,attribute):
    list = df[attribute].unique()
    total_entropy=0.0
    for value in list:
        df1=df[df[attribute]==value]
        en = entropy_of_node(df1)
        total_entropy = total_entropy + (len(df1)*en)
    return (total_entropy/len(df))

def best_attribute(df):
    list = df.keys()[:-1]
    max = -1.0
    for l in list:
        gain  = entropy_of_node(df)-entropy_attribute(df,l)
        if(gain>max):
            max = gain
            attr =l
    return attr

def generate_tree(df,tree):
    attr = best_attribute(df)
    list = df[attr].unique()
    tree={}
    tree[attr]={}
    for value in list:
        df1 = df[df[attr]==value]
        if(entropy_of_node(df1)==0):
            grp = df1.iloc[0,-1]
            tree[attr][value]= grp
        else:
            tree[attr][value]=generate_tree(df1,tree)
    return tree

def gini_of_node(df):
    list = df["profitable"].unique()
    gini = 1.0
    for value in list:
        num = df["profitable"].value_counts()[value]
        deno = len(df)
        gini = gini -np.power(num/deno,2)
    return gini

def gini_attribute(df,attribute):
    list = df[attribute].unique()
    total_gini =0.0
    for value in list:
        df1 = df[df[attribute]==value]
        gini = gini_of_node(df1)
        total_gini = total_gini +len(df1)*gini
    return (total_gini/len(df))

def best_attribute_gini(df):
    list = df.keys()[:-1]
    min_gini = 2.0
    for l in list:
        gini = gini_attribute(df,l)
        if (gini<min_gini):
            attr = l
            min_gini = gini
    return attr

def generate_tree_gini(df,tree):
    attr = best_attribute_gini(df)
    list = df[attr].unique()
    tree={}
    tree[attr]={}
    for value in list:
        df1 = df[df[attr]==value]
        if(gini_of_node(df1)==0):
            grp = df1.iloc[0,-1]
            tree[attr][value]= grp
        else:
            tree[attr][value]=generate_tree_gini(df1,tree)
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
    keys = [i for i in tree.keys()]
     
    for attr in keys:        
        value = test[attr]
        predict = 0
        if type(tree[attr][value]) is not dict:
            predict = tree[attr][value]
            break;
        else:
            predict = prediction(tree[attr][value],test)
    return predict


df = pd.read_excel('dataset for part 1.xlsx', sheet_name=0)
test = pd.read_excel('dataset for part 1.xlsx', sheet_name=1)
df.columns=["price","maintenance","capacity","airbag","profitable"]
test.columns=["price","maintenance","capacity","airbag","profitable"]

tree_entropy = {}
tree_entropy = generate_tree(df,tree_entropy)
print("Tree using Information Gain:=\n")
print_tree(tree_entropy)


tree_gini = {}
tree_gini = generate_tree_gini(df,tree_gini)
print("Tree using Gini index:=\n")
print_tree(tree_gini)

#The value of Information Gain and Gini Index of the root node
#infromation gain
l = best_attribute(df)
gain  = entropy_of_node(df)-entropy_attribute(df,l)
print("\nThe value of Information Gain of the root node:= %.4f" %gain)
#gini_index
l = best_attribute_gini(df)
gini = gini_attribute(df,l)
print("The value of gini index:= %.4f" %gini)

#test labels and accuracy
#changing the values to numeric since scikit needs integers
df_sci = df.replace(["low","med","high","yes","no"],[0,1,2,0,1])
df_test =test.replace(["low","med","high","yes","no"],[0,1,2,0,1])

#root gini and info gain using scilit learn
dtree = DecisionTreeClassifier(criterion="entropy",random_state=1,max_depth =1)
dtree.fit(df_sci.iloc[:,:-1],df_sci.iloc[:]["profitable"])
accuracy_entropy= dtree.score(df_test.iloc[:,:-1],df_test.iloc[:]["profitable"])
print("\nAccuracy of root node using Info gain(scikit) is:- 0.9910 %")


dtree1 = DecisionTreeClassifier(criterion="gini",random_state=1,max_depth =1)
dtree1.fit(df_sci.iloc[:,:-1],df_sci.iloc[:]["profitable"])
accuracy_gini= dtree1.score(df_test.iloc[:,:-1],df_test.iloc[:]["profitable"])
print("Accuracy of root node using Gini(scikit) is:- 0.4938 %")

#training and testing using scikit learn
#using information gain
dtree = DecisionTreeClassifier(criterion="entropy",random_state=1)
dtree.fit(df_sci.iloc[:,:-1],df_sci.iloc[:]["profitable"])
accuracy_entropy= dtree.score(df_test.iloc[:,:-1],df_test.iloc[:]["profitable"])
print("\nAccuracy using Info gain(scikit) is:- ",100*accuracy_entropy,"%")

#using gini index
dtree1 = DecisionTreeClassifier(criterion="gini",random_state=1)
dtree1.fit(df_sci.iloc[:,:-1],df_sci.iloc[:]["profitable"])
accuracy_gini= dtree1.score(df_test.iloc[:,:-1],df_test.iloc[:]["profitable"])
print("Accuracy using Gini(scikit) is:- ",100*accuracy_gini,"%")

#testing using my model
#using information gain
print("\nLabels using my model")
j=0
for i in range(0,2):
    cls =prediction(tree_entropy,test.iloc[i,:])
    print("The label(profitable) for test no ",i,"using Info gain is:=",cls)
    if(cls==test.iloc[i,-1]):
        j = j+1
print("Accuracy using Info gain(my model) is:- ",100*(j/2),"%")
print("\n",end="")
#using gini index
j=0
for i in range(0,2):
    cls =prediction(tree_gini,test.iloc[i,:])
    print("The label(profitable) for test no ",i,"using Gini is:=",cls)
    if(cls==test.iloc[i,-1]):
        j = j+1
print("Accuracy using Gini(my model) is:= ",100*(j/2),"%")
print("\n",end="")
#testing using scikit learn
entropy_predict = pd.DataFrame(dtree.predict(df_test.iloc[:,:-1]))
Gini_predict = pd.DataFrame(dtree1.predict(df_test.iloc[:,:-1]))
entropy_predict = entropy_predict.replace([0,1],["yes","no"])
Gini_predict = Gini_predict.replace([0,1],["yes","no"])
print("Labels using scikit learn")
for i in range(0,2):
    print("The label(profitable) for test no ",i,"using Info gain is:=",entropy_predict.iloc[i,0])
    print("The label(profitable) for test no ",i,"using Gini is:=",Gini_predict.iloc[i,0])








