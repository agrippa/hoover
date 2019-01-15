#!/usr/bin/python

import os
import sys
import glob

if len(sys.argv) != 2:
    sys.stderr.write('usage: python check_shmalloc_traces.py <job-dir>\n')
    sys.exit(1)

job_dir = sys.argv[1]
files = glob.glob(job_dir + '/shmem_malloc_*.txt')

all_allocs = {}
for filename in files:
    fp = open(filename, 'r')
    content = fp.read()
    fp.close()
    lines = content.strip().split('\n')
    allocs = []
    pe = -1
    for l in lines:
        tokens = l.split()
        pe = int(tokens[0])
        allocs.append(int(tokens[2]))
    assert pe >= 0
    all_allocs[pe] = allocs

print('Found ' + str(len(all_allocs.keys())) + ' PEs')
ref = all_allocs[0]
for pe in range(1, len(all_allocs.keys())):
    comp = all_allocs[pe]
    assert len(ref) == len(comp)
    for i in range(len(ref)):
        assert ref[i] == comp[i]
print('Checks passed!')
