#!/usr/bin/python

import os
import sys

def unique_id_from_vert(vert, min_attr, max_attr):
    return tuple(vert[min_attr:max_attr + 1])

def read_vertex_trace_file(fp, min_id_attr, max_id_attr):
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

        uid = unique_id_from_vert(vec, min_id_attr, max_id_attr)
        assert uid not in records
        records[uid] = {'iter': it, 'pe': pe, 'vec_id': vec_id, 'vec': vec}
    return records

def vertices_are_equal(r1, r2):
    vec1 = r1['vec']
    vec2 = r2['vec']

    assert len(vec1) == len(vec2)
    for i in range(len(vec1)):
        if vec1[i] != vec2[i]:
            return False
    return True

def vertex_to_str(r):
    vec_id = r['vec_id']
    offset = vec_id & 0xffffffff
    pe = vec_id >> 32
    return '{PE=' + str(r['pe']) + ', ID=' + str(vec_id) + ' (PE=' + str(pe) + \
            ', OFFSET=' + str(offset) + '), VEC=' + str(r['vec']) + '}'

if __name__ == '__main__':
    if len(sys.argv) != 5:
        sys.stderr.write('usage: python compare_traces.py file1 file2 ' +
                'min-id-attr max-id-attr\n')
        sys.exit(1)

    fp1 = open(sys.argv[1], 'r')
    fp2 = open(sys.argv[2], 'r')
    min_id_attr = int(sys.argv[3])
    max_id_attr = int(sys.argv[4])

    records1 = read_vertex_trace_file(fp1, min_id_attr, max_id_attr)
    print('Loaded ' + str(len(records1)) + ' records from ' + sys.argv[1])
    records2 = read_vertex_trace_file(fp2, min_id_attr, max_id_attr)
    print('Loaded ' + str(len(records2)) + ' records from ' + sys.argv[2])

    for r1_id in records1.keys():
        r1 = records1[r1_id]
        r2 = records2[r1_id]

        if not vertices_are_equal(r1, r2):
            print(vertex_to_str(r1))
            print(vertex_to_str(r2))
            print('')
    print('Done!')

