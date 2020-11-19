#!/usr/bin/env python3

# take the modified cell_hierarchy.csv file, extract 
# the relevant data, sort by cell ID, and write new file
#
# usage: rt-debugging-gradient-transport.py cell_hierarchy.csv run
#       cell_hierarchy.csv: manually modified cell output form swift
#       run:    integer, run number for which to work with. Will be
#               added to resulting filename

import numpy as np
import sys

fname = sys.argv[1]
run = sys.argv[2] # keep it a string
#  test = int(run)

skip_empties = True


cols = [0, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 
        31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45]

data = np.loadtxt(fname, skiprows = 2, usecols=cols, dtype=np.int64, delimiter=",")
sortind = data[:,0].argsort()
data = data[sortind]



newfname = fname[:-4]+"_sorted_"+ run + ".csv"

newfp = open(newfname, "w")


linelen = 18 + 4 + (3*15 + 7 + 3*3 + 4) + (3*15 + 7 + 3*3 + 3)
newfp.write("{0:18} || {1:61} || {2:61} ||\n".format("CellID", "gradient", "transport"))
newfp.write("{0:18} || {1:15} | {2:15} | {3:15} | {0:7} || {1:15} | {2:15} | {3:15} | {0:7} ||\n".format( 
                " ", "created", "cell_unskip", "engine_marktask"))

for l in range(linelen):
    newfp.write("-")
newfp.write("\n")

taskline = "{0:18} || ".format(" ")
for i in range(2):
    for i in range(3):
        taskline += "{0:3} {1:3} {2:3} {3:3} | ".format("S", "SS", "P", "SP")
    taskline += "{0:3} {1:3} || ".format("LA", "LW")
taskline+="\n"
newfp.write(taskline)

for l in range(linelen):
    newfp.write("=")
newfp.write("\n")


for i in range(data.shape[0]):
    if (data[i, 1:] == 0).all() and skip_empties:
        # don't write cells where everything is = 0
        continue
    newfp.write(("{0:18d} || "+
                "{1:3d} {2:3d} {3:3d} {4:3d} | {5:3d} {6:3d} {7:3d} {8:3d} | "+
                "{9:3d} {10:3d} {11:3d} {12:3d} | {13:3d} {14:3d} || "+
                "{15:3d} {16:3d} {17:3d} {18:3d} | {19:3d} {20:3d} {21:3d} {22:3d} | "+
                "{23:3d} {24:3d} {25:3d} {26:3d} | {27:3d} {28:3d} ||\n").format(
                    data[i,0], 
                    data[i,1], data[i,2], data[i,3], data[i,4], data[i,5], data[i,6], data[i,7], data[i,8], 
                    data[i,9], data[i,10], data[i,11], data[i,12], data[i,13], data[i,14], 
                    data[i,15], data[i,16], data[i,17], data[i,18], data[i,19], data[i,20], data[i,21], data[i,22], 
                    data[i,23], data[i,24], data[i, 25], data[i,26], data[i,27], data[i,28]
                ))

    if (data[i,1:15] != data[i, 15:]).any():
        print("Found grad/transport difference in cell", data[i,0])


newfp.close()
