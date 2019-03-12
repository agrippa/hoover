#!/usr/bin/python

import os
import sys

def read_trace_file(fp):
    records = {}
    for line in fp:
        tokens = line.split(',')
        it = int(tokens[0])
        pe = int(tokens[1])
        vec_id = int(tokens[2])
        nelements = int(tokens[3])

        vec = []
        vals = tokens[4:]
        assert len(vals) % 2 == 0
        assert int(len(vals) / 2) == nelements
        for i in range(int(len(vals) / 2)):
            assert int(vals[2 * i]) == i
            vec.append(float(vals[2 * i + 1]))
        assert vec_id not in records
        records[vec_id] = {'iter': it, 'pe': pe, 'vec_id': vec_id, 'vec': vec}
    return records

def is_equal(r1, r2):
    vec1 = r1['vec']
    vec2 = r2['vec']

    assert len(vec1) == len(vec2)
    for i in range(len(vec1)):
        if vec1[i] != vec2[i]:
            return False
    return True

def record_to_str(r):
    return '{PE=' + str(r['pe']) + ', ID=' + str(r['vec_id']) + ', VEC=' + \
            str(r['vec']) + '}'

if len(sys.argv) != 3:
    sys.stderr.write('usage: python compare_traces.py file1 file2\n')
    sys.exit(1)

fp1 = open(sys.argv[1], 'r')
fp2 = open(sys.argv[2], 'r')

records1 = read_trace_file(fp1)
print('Loaded ' + str(len(records1)) + ' records from ' + sys.argv[1])
records2 = read_trace_file(fp2)
print('Loaded ' + str(len(records2)) + ' records from ' + sys.argv[2])

for r1_id in records1.keys():
    r1 = records1[r1_id]
    r2 = records2[r1_id]

    if not is_equal(r1, r2):
        print(record_to_str(r1))
        print(record_to_str(r2))
        print('')

