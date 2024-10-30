#!/usr/bin/env python3

import sys
import random
import struct
import string

def printFloat(floatn, prec, PLUSIGN=True):
    param = '{:.' + str(prec) + 'e}'
    plus = ""
    if (floatn > 0) and PLUSIGN:
        plus = "+"
    print(plus+param.format(floatn), end="")

def RandomFloat(typef):
    if typef=="double":
        return struct.unpack('d', random.randbytes(8))[0] 
    elif typef=="float":
        return struct.unpack('f', random.randbytes(4))[0] 
    elif typef=="_Float16":
        return struct.unpack('e', random.randbytes(2))[0]
    elif typef=="__bf16":
        return struct.unpack('e', random.randbytes(2))[0]

def nonNanRandomFloat(typef):
    val = RandomFloat(typef)
    while (val != val): #test for nan
        val = RandomFloat(typef)
    return val

def positiveNonNanFloat(typef):
    return abs(nonNanRandomFloat(typef))

def worstCaseSimple(N):
    L = []
    L.append(1.0)
    for i in range(N-1):
        L.append(1.1102230246251565E-16) #2**-53
    return L

if __name__=='__main__':
    try:
        N = int(sys.argv[1])
    except IndexError:
        N = 100

    try:
        prec = int(sys.argv[2])
    except IndexError:
        prec = 16

    try:
        n = int(sys.argv[3])
    except IndexError:
        n = 1

    try:
        typef = sys.argv[4]
    except IndexError:
        typef = "float" 
    
    try:
        name = sys.argv[5]
    except IndexError:
        name = ''.join(random.sample(string.ascii_lowercase,5))

    arrayname = typef+"_"+str(N)+"p"+str(prec)+"_"+name
    """ <-- add or remove # to toggle
    #''' <-- add or remove # to toggle
    #unsorted
    print(typef+" "+arrayname+"["+str(N)+"] = {", end="")
    for i in range(N):
        if not((i % n)): #return line
            print("\n",end="") 
        floatval = positiveNonNanFloat(typef)
        #floatval = nonNanRandomFloat(typef)
        printFloat(floatval, prec, PLUSIGN=False)
        print(",",end="\t")
    print("\n};")
    ''' 
    #sorted
    floatList = [positiveNonNanFloat(typef) for i in range(N)]
    print(typef+" "+arrayname+"["+str(N)+"] = {", end="")
    for i in range(N):
        if not((i % n)): #return line
            print("\n",end="")
        printFloat(floatList[i], prec, PLUSIGN=False)
        print(",",end="\t")
    print("\n};\n")
    floatList.sort()
    print(typef+" "+arrayname+"_sorted["+str(N)+"] = {", end="")
    for i in range(N):
        if not((i % n)): #return line
            print("\n",end="")
        printFloat(floatList[i], prec, PLUSIGN=False)
        print(",",end="\t")
    print("\n};\n")
    floatList.reverse()
    print(typef+" "+arrayname+"_sorted_rev["+str(N)+"] = {", end="")
    for i in range(N):
        if not((i % n)): #return line
            print("\n",end="")
        printFloat(floatList[i], prec, PLUSIGN=False)
        print(",",end="\t")
    print("\n};")
    #'''
    """
    repeat = 4
    worst = worstCaseSimple(N)
    print("double worstCaseSimple_"+str(N*repeat)+
          "["+str(N*repeat)+"] = {", end="")
    for k in range(repeat):
        for i in range(N):
            if not((i % n)): #return line
                print("\n",end="") 
            printFloat(worst[i] , prec, PLUSIGN=False)
            print(",",end="\t")
    print("\n};\n")
    #"""
    exit(0) 
