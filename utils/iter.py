#!/usr/bin/env python3
'''
plot iter graphs using matplotlib
'''

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import csv
import sys
import os

#"""
USE_BASE2=False
"""
USE_BASE2=True
#"""

SIGNIFICANTD_MODE = False

def stripname(name):
    st = name.split("_")
    return st[-3] + " " + st[-2]   

def getstepiter(name):
    st = name.split("_")
    return int(st[-1]) 

def titlename(name, ran):
    st = name.split("_")
    typef=st[0]
    if typef=='':
        typef=st[1] 
    return "(reduction on "+str(ran*getstepiter(l))+" "+typef+")"   

if __name__ == "__main__":
    if USE_BASE2:
        X = 2
    else:
        X = 10

    try:
        csvdir = os.path.expandvars(sys.argv[1])
        
        if (csvdir.find("SIGNIFICANTD") != -1):
            SIGNIFICANTD_MODE = True
        
        bottom = int(sys.argv[2])
        top = int(sys.argv[3])
        try:
            ran = int(sys.argv[4])#how much value tu see
            off = int(sys.argv[5])#offset to view data
            RANGE_SET = True
        except IndexError:
            RANGE_SET = False
            ran = 0
            off = 0
    except IndexError:
        print("Error : Missing arguments.\nUsage :\
                \n\titer.py \"path/to/csv/dir\" bottom_lim top_lim [range offset]")
        exit(1)
    os.chdir(csvdir)

    fig, ax = plt.subplots()
    for l in os.listdir():
        if "png" in l:
            continue
        print(l)
        with open(l, 'r') as f:
            data = [float(line.rstrip(',\n')) for line in f]
            #print(data)
        if not(RANGE_SET):
            ran = len(data)
        xax = np.arange((1+off)*getstepiter(l), (ran+off+1)*getstepiter(l)
            , getstepiter(l))
        ax.grid()
        #print(data[off:ran+off])
        tmp = np.array(data[off:ran+off])
        filtered = np.where(tmp < (X**top), tmp, None)
        ax.plot(xax, filtered
                , marker='x', markersize=3, label=stripname(l), linewidth=0.6)

    def forward(x):
        if USE_BASE2:
            return -(np.log2(x)-top)**2
        else:
            return -(np.log10(x)-top)**2
                                                                                
    def inverse(x):                                                             
        return X**((-x)**(1/2) + top) # X is either 2 or 10

    ax.margins(y=0)
    
    i = bottom
    arr = []                                                                    
    while (i < top+1):
        if SIGNIFICANTD_MODE:
            arr.append(i)
        else:
            arr.append(X ** i)                                                    
        i+=1
    print(arr)
    
    ax.set_ylim(arr[0], arr[1])
    print(ax.get_ylim())

    if not(SIGNIFICANTD_MODE):
        ax.set_yscale('function', functions=(forward, inverse))
        if USE_BASE2:
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.5f'))
        else:
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0e'))
    
    ax.set_yticks(arr)

    if SIGNIFICANTD_MODE:
        ax.set(ylabel='Number of significantd', xlabel='iteration'
            , title='Number of significantd (compared to fp64 result)\
 evolution comparison '+ titlename(l,ran))
        ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        plt.savefig("iterations.png", format="png", bbox_inches="tight"
                    , pad_inches=0.2)
    else:
        ax.set(ylabel='Distance to fp64 reduction', xlabel='iteration'
            , title='Error (to fp64 result) evolution comparison '
               + titlename(l,ran))
        ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        plt.savefig("iterations.png", format="png", bbox_inches="tight"
                    , pad_inches=0.2)
    plt.show()
