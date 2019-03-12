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
        n_neighbors = int(tokens[3])

        vals = tokens[4:]
        assert len(vals) == n_neighbors

        edges = []
        for e in vals:
            t = e.split(':')
            assert len(t) == 3
            edges.append({'dir': t[0],
                          'pe': int(t[1]),
                          'offset': int(t[2])})

        assert vec_id not in records
        records[vec_id] = {'iter': it,
                           'pe': pe,
                           'vec_id': vec_id,
                           'edges': edges}
    return records

def is_equal(r1, r2):
    edges1 = r1['edges']
    edges2 = r2['edges']

    if len(edges1) != len(edges2):
        return False

    for e in edges1:
        target_pe = e['pe']
        target_offset = e['offset']

        found = None
        for e2 in edges2:
            if e2['pe'] == target_pe and e2['offset'] == target_offset:
                found = e2
        if found is None or found['dir'] != e['dir']:
            return False
    return True

def record_to_str(r):
    return '{PE=' + str(r['pe']) + ', ID=' + str(r['vec_id']) + ', EDGES=' + \
            str(r['edges']) + '}'

if len(sys.argv) != 3:
    sys.stderr.write('usage: python compare_edge_traces.py file1 file2\n')
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

