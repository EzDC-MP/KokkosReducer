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
    try:
        csvdir = os.path.expandvars(sys.argv[1])
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
        ax.plot(xax, data[off:ran+off]
                , marker='x', markersize=3, label=stripname(l), linewidth=0.6)
         
    def forward(x):                                                             
        return -(np.log10(x)-top)**2                  
                                                                                
    def inverse(x):                                                             
        return 10**((-x)**(1/2) + top)

    ax.margins(y=0)
    
    i = bottom
    arr = []                                                                    
    while (i < top+1):
        arr.append(10 ** i)                                                    
        i+=1
    print(arr)
    
    ax.set_ylim(arr[0], arr[1])
    print(ax.get_ylim())
    ax.set_yscale('function', functions=(forward, inverse))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0e'))             
    ax.set_yticks(arr)


    ax.set(ylabel='Distance to fp64 reduction', xlabel='iteration'
            , title='Error (to fp64 result) evolution comparaison '+ titlename(l,ran))
    ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.savefig("iterations.png", format="png", bbox_inches="tight", pad_inches=0.2)
    plt.show()
