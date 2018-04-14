#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 14:45:51 2018

@author: David
"""
from graphviz import Digraph

def assembleKnights():
    roundTable = Digraph(comment="The Round Table")
    roundTable.node('A', 'King Arthur')
    roundTable.node('B', 'Sir Bedevere the Wise')
    roundTable.node('L', 'Sir Lancelot the Brave')
    roundTable.edges(['AB', 'AL'])
    roundTable.edge('B', 'L', constraint='false')
    return roundTable

def main():
    x = "hello world"
    print(x)
    
    knights = assembleKnights()
    print(knights.source)

if __name__ == "__main__":
    main()