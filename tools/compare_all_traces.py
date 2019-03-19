#!/usr/bin/python

import os
import sys

from compare_traces import read_vertex_trace_file, vertices_are_equal, \
        vertex_to_str
from compare_edge_traces import read_edge_trace_file, edges_are_equal, \
        edges_to_str

if __name__ == '__main__':
    if len(sys.argv) != 5:
        sys.stderr.write('usage: python compare_all_traces.py vertex-trace-1 ' +
                'vertex-trace-2 edge-trace-1 edge-trace-2\n')
        sys.exit(1)

    vertex_fp1 = open(sys.argv[1], 'r')
    vertex_fp2 = open(sys.argv[2], 'r')
    edge_fp1 = open(sys.argv[3], 'r')
    edge_fp2 = open(sys.argv[4], 'r')

    vertices1 = read_vertex_trace_file(vertex_fp1)
    print('Loaded ' + str(len(vertices1)) + ' vertices from ' + sys.argv[1])
    vertices2 = read_vertex_trace_file(vertex_fp2)
    print('Loaded ' + str(len(vertices2)) + ' vertices from ' + sys.argv[2])
    edges1 = read_edge_trace_file(edge_fp1)
    print('Loaded edge info for ' + str(len(edges1)) + ' vertices from ' +
            sys.argv[3])
    edges2 = read_edge_trace_file(edge_fp2)
    print('Loaded edge info for ' + str(len(edges2)) + ' vertices from ' +
            sys.argv[4])
    print('')

    for vid in vertices1.keys():
        assert vid in vertices2
        assert vid in edges1
        assert vid in edges2

        vertex_equal = vertices_are_equal(vertices1[vid], vertices2[vid])
        edges_equal = edges_are_equal(edges1[vid], edges2[vid])

        if not vertex_equal or not edges_equal:
            print('Vertex equal? ' + str(vertex_equal) + ', edges equal? ' +
                    str(edges_equal))
            print(vertex_to_str(vertices1[vid]))
            print(vertex_to_str(vertices2[vid]))
            print(edges_to_str(edges1[vid]))
            print(edges_to_str(edges2[vid]))

