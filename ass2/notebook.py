#!/usr/bin/env python
# coding: utf-8

# In[2]:


# %load ass2p1.py
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import pprint
df = pd.read_excel('dataset for part 1.xlsx', sheet_name=0)
test = pd.read_excel('dataset for part 1.xlsx', sheet_name=1)
df.columns=["price","maintenance","capacity","airbag","profitable"]
test.columns=["price","maintenance","capacity","airbag","profitable"]
print(df)


# In[3]:


def entropy_of_node(df):
    list=df["profitable"].unique()
    entropy = 0.0
    for value in list:
        num=df["profitable"].value_counts()[value]
        deno = len(df)
        p=(num/deno)
        entropy = entropy + (-p*(np.log2(p)))
    return entropy

print(entropy_of_node(df))
    


# In[4]:


def entropy_attribute(df,attribute):
    list = df[attribute].unique()
    total_entropy=0.0
    for value in list:
        df1=df[df[attribute]==value].reset_index(drop=True)
        en = entropy_of_node(df1)
        total_entropy = total_entropy + (len(df1)*en)
    return (total_entropy/len(df))


# In[27]:


df1 = df[df["profitable"]=="yes"].reset_index(drop=True)
print(df1)
print(df1.iloc[2,2])
print(entropy_of_node(df1))
print(df1.iloc[0,-1])
l=df["profitable"].unique()
print(l)
z = np.log2(df.loc[0]["capacity"])
(z)


# In[5]:


def best_attribute(df):
    list = df.keys()[:-1]
    max = -1.0
    for l in list:
        gain  = entropy_of_node(df)-entropy_attribute(df,l)
        if(gain>max):
            max = gain
            attr =l
    return attr

print(best_attribute(df))


# In[6]:


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

tree = {}
tree = generate_tree(df,tree)
pprint.pprint(tree,width=1)


# In[7]:


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

print_tree(tree,0)


# In[9]:


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

print(prediction(tree,test.iloc[1,:]))
print(prediction(tree,test.iloc[0,:]))


# In[10]:


def gini_of_node(df):
    list = df["profitable"].unique()
    gini = 1.0
    for value in list:
        num = df["profitable"].value_counts()[value]
        deno = len(df)
        gini = gini -np.power(num/deno,2)
    return gini

gini_of_node(df)


# In[12]:


def gini_attribute(df,attribute):
    list = df[attribute].unique()
    total_gini =0.0
    for value in list:
        df1 = df[df[attribute]==value]
        gini = gini_of_node(df1)
        total_gini = total_gini +len(df1)*gini
    return (total_gini/len(df))

print(gini_attribute(df,"maintenance"))


# In[13]:


def best_attribute_gini(df):
    list = df.keys()[:-1]
    min_gini = 2.0
    for l in list:
        gini = gini_attribute(df,l)
        if (gini<min_gini):
            attr = l
            min_gini = gini
    return attr

best_attribute_gini(df)


# In[14]:


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
            tree[attr][value]=generate_tree(df1,tree)
    return tree

tree1 = {}
tree1 = generate_tree(df,tree1)
pprint.pprint(tree1,width=1)


# In[15]:


df_sci = df.replace(["low","med","high","yes","no"],[0,1,2,0,1])
df_test =test.replace(["low","med","high","yes","no"],[0,1,2,0,1])
print(df_sci)
dtree = DecisionTreeClassifier(criterion="entropy",random_state=1)
dtree.fit(df_sci.iloc[:,:-1],df_sci.iloc[:]["profitable"])
dtree.score(df_test.iloc[:,:-1],df_test.iloc[:]["profitable"])


# In[ ]:




