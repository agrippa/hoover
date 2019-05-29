#!/usr/bin/python

import os
import sys

def edge_key(e):
    if e['dir'] == 'IN':
        return (-1, e['pe'], e['offset'])
    elif e['dir'] == 'BIDIRECTIONAL':
        return (0, e['pe'], e['offset'])
    elif e['dir'] == 'OUT':
        return (1, e['pe'], e['offset'])
    else:
        assert False


def read_edge_trace_file(fp):
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

        edges.sort(key = edge_key)

        assert vec_id not in records
        records[vec_id] = {'iter': it,
                           'pe': pe,
                           'vec_id': vec_id,
                           'edges': edges}
    return records

def find_vertex_for_vecid(vecid, vert_dict):
    for uid in vert_dict.keys():
        if vert_dict[uid]['vec_id'] == vecid:
            return uid
    return None

def edges_are_equal(r1, r2, vert_dict_1, vert_dict_2, vecid_to_uid_1,
        vecid_to_uid_2):
    edges1 = r1['edges']
    edges2 = r2['edges']

    if len(edges1) != len(edges2):
        return False

    for e in edges1:
        target_pe = e['pe']
        target_offset = e['offset']
        vecid1 = (target_pe << 32) + target_offset
        uid1 = vecid_to_uid_1[vecid1]

        found = None
        for e2 in edges2:
            vecid2 = (e2['pe'] << 32) + e2['offset']
            uid2 = vecid_to_uid_2[vecid2]
            if uid1 == uid2:
                found = e2
        if found is None or found['dir'] != e['dir']:
            return False
    return True

def uid_to_vertex_str(uid, vertices):
    vert = vertices[uid]
    return str(uid) + ' : ' + str(vert['vec'])


def edges_to_str(r, vecid_to_uid, uid_to_vertices):
    edge_str = '\n'.join(['    ' + str(e['dir']) + ' ' +
        uid_to_vertex_str(vecid_to_uid[(e['pe'] << 32) + e['offset']], uid_to_vertices) for e in r['edges']])
    return '{PE=' + str(r['pe']) + ', ID=' + str(r['vec_id']) + ', EDGES=\n' + \
            edge_str + '}'
