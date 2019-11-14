#!/usr/bin/python

import os
import sys

d = {}
elapsed_time = None
npes = None

fp = open(sys.argv[1], 'r')
line_no = 1
try:
    for line in fp:
        # PE 6 found 2 patterns on timestep 207 using 5664 visits, 5542 local vertices in total. Best score = 164, vertex count = 2, edge count = 1. # local gets = 5828, # remote gets = 0
        if line.find('local vertices in total') != -1:
            tokens = line.split()
            pe = int(tokens[1])
            verts = int(tokens[29])
            visits = int(tokens[9])
            if not pe in d:
                d[pe] = []
            d[pe].append((verts, visits))
        elif line.find('total CPU') != -1:
            tokens = line.split()
            elapsed_time = float(tokens[11])
        elif line.find('PE(s)') != -1:
            npes = int(line.split()[0])
        line_no += 1
except Exception as e:
    sys.stderr.write('Error on line ' + str(line_no) + '\n')
    raise e

s = 0
for pe in d:
    s += d[pe][-1][0]
total_visits = 0
for pe in d:
    for it in d[pe]:
        total_visits += it[1]
print(str(s) + ' vertices added in total')
print(str(total_visits) + ' visits in total')
print(str(elapsed_time) + ' ms elapsed (wallclock)')
print(str(1000.0 * float(s) / elapsed_time) + ' verts / s')
print(str(1000.0 * float(total_visits) / elapsed_time) + ' visits / s')
print(str(npes) + ' PEs, ' + str(npes / 32) + ' nodes assuming 32 PEs per node')
