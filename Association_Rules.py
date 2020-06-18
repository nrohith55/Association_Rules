# -*- coding: utf-8 -*-
"""
Created on Wed May 27 15:51:22 2020

@author: Rohith
"""

import pandas as pd

groceries=[]

with open('E:\\Data Science\\Assignments\\Python code\\Association_Rules\\groceries.csv') as f:
    groceries=f.read()

groceries=groceries.split("\n")

groceries_list=[]

for i in groceries:
    groceries_list.append(i.split(","))


print(groceries_list)

all_groceries_list=[i for items in groceries_list for i in items]

from collections import Counter
item_frequencies = Counter(all_groceries_list)

item_frequencies = sorted(item_frequencies.items(),key = lambda x:x[1])
frequencies = list(reversed([i[1] for i in item_frequencies]))
items = list(reversed([i[0] for i in item_frequencies]))


#Bar plot


import matplotlib.pyplot as plt
plt.bar(x=list(range(0,11)) ,height=frequencies[0:11],color='rgbkymc');
plt.xticks(list(range(0,11),),items[0:11]);
plt.xlabel("items")
plt.ylabel("Count");plt.xlabel("Items")
# Creating Data Frame for the transactions data
g_series=pd.DataFrame(pd.Series(groceries_list))
g_series=g_series.iloc[:9835,:]
g_series.columns=["Transcations"]


# Creating Data Frame for the transactions data
X = g_series['Transcations'].str.join(sep='*').str.get_dummies(sep='*')

from mlxtend.frequent_patterns import apriori,association_rules

help(apriori)

frequent_items=apriori(X, min_support=0.005, max_len=3,use_colnames = True)

# Most Frequent item sets based on support 
frequent_items.sort_values('support',ascending=False,inplace=True)

plt.bar(x = list(range(1,11)),height = frequent_items.support[1:11],color='rgmyk');
plt.xticks(list(range(1,11)),frequent_items.itemsets[1:11])
plt.xlabel('item-sets');plt.ylabel('support')

help(association_rules)
rules=association_rules(frequent_items,metric='lift', min_threshold=1)

rules.head(20)
rules.sort_values('lift',ascending = False).head(10)


########################################################################################























