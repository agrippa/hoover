#!/usr/bin/python

import os
import sys


def parse_line(timestep, tokens):
    d = {}
    d['timestep'] = timestep
    d['total'] = float(tokens[4])
    d['metadata'] = float(tokens[8])
    d['summary'] = float(tokens[14])
    d['edges'] = float(tokens[18])
    d['getmem'] = float(tokens[20][1:])
    d['update_edges'] = float(tokens[21][:len(tokens[21]) - 1])
    d['neighbor_updates'] = float(tokens[25])
    d['abort'] = float(tokens[29])
    d['nspins'] = int(tokens[32])
    d['nneighbors'] = int(tokens[35])
    return d

if len(sys.argv) != 3:
    print('usage: python extract_perf_data.py <log-file> <metric>')
    sys.exit(1)

initial_timestep = 2
pe_timesteps = {}
pe_data = {}

fp = open(sys.argv[1], 'r')
for line in fp:
    line = line.strip()

    if line.startswith('PE '):
        tokens = line.split()
        if len(tokens) >= 4 and tokens[3] == 'total':
            pe = int(tokens[1])

            if not pe in pe_timesteps:
                pe_timesteps[pe] = initial_timestep
                pe_data[pe] = []

            data = parse_line(pe_timesteps[pe], tokens)
            pe_data[pe].append(data)
            pe_timesteps[pe] = pe_timesteps[pe] + 1

fp.close()

npes = len(pe_data)
ntimesteps = len(pe_data[0])

for t in range(ntimesteps):
    s = str(pe_data[0][t]['timestep'])
    for p in range(npes):
        s += ',' + str(pe_data[p][t][sys.argv[2]])
    s += ',,'
    print(s)

