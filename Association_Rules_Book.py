# -*- coding: utf-8 -*-
"""
Created on Wed May 27 16:41:53 2020

@author: Rohith
"""



import pandas as pd

X=pd.read_csv("E:\\Data Science\\Assignments\\Python code\\Association_Rules\\book.csv")

from mlxtend.frequent_patterns import apriori,association_rules
frequent_items=apriori(X, min_support=0.005, max_len=3,use_colnames = True)

frequent_items.sort_values('support',ascending=False,inplace=True)

rules=association_rules(frequent_items,metric='lift', min_threshold=1)

print(rules.sort_values('lift',ascending=False).head())

import matplotlib.pyplot as plt

plt.bar(x = list(range(1,11)),height = frequent_items.support[1:11],color='rgmyk');
plt.xticks(list(range(1,11)),frequent_items.itemsets[1:11])
plt.xlabel('item-sets');plt.ylabel('support')