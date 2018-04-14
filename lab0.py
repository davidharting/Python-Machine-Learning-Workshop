#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 14:45:51 2018

@author: David
"""
from graphviz import Digraph

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def assembleKnights():
    roundTable = Digraph(comment="The Round Table")
    roundTable.node('A', 'King Arthur')
    roundTable.node('B', 'Sir Bedevere the Wise')
    roundTable.node('L', 'Sir Lancelot the Brave')
    roundTable.edges(['AB', 'AL'])
    roundTable.edge('B', 'L', constraint='false')
    return roundTable

def plot():
    """
    This example is just straight up copy-pasted from
    https://matplotlib.org/gallery/lines_bars_and_markers/simple_plot.html
    I really just wanted to make sure my environment was correctly set up
    to render matplotlib plots, and it is!
    """

    # Data for plotting
    t = np.arange(0.0, 2.0, 0.01)
    s = 1 + np.sin(2 * np.pi * t)

    # Note that using plt.subplots below is equivalent to using
    # fig = plt.figure() and then ax = fig.add_subplot(111)
    fig, ax = plt.subplots()
    ax.plot(t, s)

    ax.set(xlabel='time (s)', ylabel='voltage (mV)',
        title='About as simple as it gets, folks')
    ax.grid()

    fig.savefig("output/plot.png")

def main():
    x = "hello world"
    print(x)
    
    knights = assembleKnights()
    print(knights.source)
    knights.render('output/round-table.gv', view=False)

    plot()

if __name__ == "__main__":
    main()
