#!/usr/bin/python

import os
import sys

d = {}
elapsed_time = None
npes = None

fp = open(sys.argv[1], 'r')
line_no = 1
for line in fp:
    # PE 6 found 2 patterns on timestep 207 using 5664 visits, 5542 local vertices in total. Best score = 164, vertex count = 2, edge count = 1. # local gets = 5828, # remote gets = 0
    if line.find('local vertices in total') != -1:
        tokens = line.split()
        pe = int(tokens[1])
        verts = int(tokens[30])
        if not pe in d:
            d[pe] = []
        d[pe].append(verts)
    elif line.find('total CPU') != -1:
        tokens = line.split()
        elapsed_time = float(tokens[11])
    elif line.find('PE(s)') != -1:
        npes = int(line.split()[0])
    line_no += 1

s = 0
for pe in d:
    s += d[pe][-1]
print(str(s) + ' vertices added in total')
print(str(elapsed_time) + ' ms elapsed (wallclock)')
print(str(1000.0 * float(s) / elapsed_time) + ' verts / s')
print(str(npes) + ' PEs, ' + str(npes / 24) + ' nodes assuming 24 PEs per node')
