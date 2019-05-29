#!/usr/bin/python

import os
import sys

from compare_traces import read_vertex_trace_file, vertices_are_equal, \
        vertex_to_str
from compare_edge_traces import read_edge_trace_file, edges_are_equal, \
        edges_to_str

if __name__ == '__main__':
    if len(sys.argv) != 7:
        sys.stderr.write('usage: python compare_all_traces.py vertex-trace-1 ' +
                'vertex-trace-2 edge-trace-1 edge-trace-2 min-id-attr ' +
                'max-id-attr\n')
        sys.exit(1)

    vertex_fp1 = open(sys.argv[1], 'r')
    vertex_fp2 = open(sys.argv[2], 'r')
    edge_fp1 = open(sys.argv[3], 'r')
    edge_fp2 = open(sys.argv[4], 'r')
    min_id_attr = int(sys.argv[5])
    max_id_attr = int(sys.argv[6])

    vertices1 = read_vertex_trace_file(vertex_fp1, min_id_attr, max_id_attr)
    print('Loaded ' + str(len(vertices1)) + ' vertices from ' + sys.argv[1])
    vertices2 = read_vertex_trace_file(vertex_fp2, min_id_attr, max_id_attr)
    print('Loaded ' + str(len(vertices2)) + ' vertices from ' + sys.argv[2])
    edges1 = read_edge_trace_file(edge_fp1)
    print('Loaded edge info for ' + str(len(edges1)) + ' vertices from ' +
            sys.argv[3])
    edges2 = read_edge_trace_file(edge_fp2)
    print('Loaded edge info for ' + str(len(edges2)) + ' vertices from ' +
            sys.argv[4])
    print('')

    vecid_to_uid_1 = {}
    for uid in vertices1.keys():
        vert = vertices1[uid]
        vecid_to_uid_1[vert['vec_id']] = uid
    vecid_to_uid_2 = {}
    for uid in vertices2.keys():
        vert = vertices2[uid]
        vecid_to_uid_2[vert['vec_id']] = uid

    for vid in vertices1.keys():
        assert vid in vertices2

        vert1 = vertices1[vid]
        vert2 = vertices2[vid]

        assert vert1['vec_id'] in edges1
        assert vert2['vec_id'] in edges2

        edge1 = edges1[vert1['vec_id']]
        edge2 = edges2[vert2['vec_id']]

        vertex_equal = vertices_are_equal(vert1, vert2)
        edges_equal = edges_are_equal(edge1, edge2, vertices1, vertices2,
                vecid_to_uid_1, vecid_to_uid_2)

        if not vertex_equal or not edges_equal:
            print('Vertex equal? ' + str(vertex_equal) + ', edges equal? ' +
                    str(edges_equal))
            print(vertex_to_str(vertices1[vid]))
            print(vertex_to_str(vertices2[vid]))
            print(edges_to_str(edge1, vecid_to_uid_1, vertices1))
            print(edges_to_str(edge2, vecid_to_uid_2, vertices2))

