#!/usr/bin/python

import os
import sys

if len(sys.argv) != 3:
    sys.stderr.write('usage: python generate_intrusion_perf_stats.py ' +
            '<job-output-file> <pes-per-node>\n')
    sys.exit(1)

d = {}
elapsed_time = None
npes = None

pes_per_node = int(sys.argv[2])
fp = open(sys.argv[1], 'r')
line_no = 1
for line in fp:
    try:
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
    except Exception as e:
        sys.stderr.write('Error on line ' + str(line_no) + '\n')
        raise e
    line_no += 1

s = 0
for pe in d:
    s += d[pe][-1]
print(str(s) + ' vertices added in total')
print(str(elapsed_time) + ' ms elapsed (wallclock)')
print(str(1000.0 * float(s) / elapsed_time) + ' verts / s')
print(str(npes) + ' PEs, ' + str(npes / pes_per_node) + ' nodes')
