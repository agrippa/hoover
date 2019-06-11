import os
import sys

if len(sys.argv) != 2:
    sys.stderr.write("usage: python summarize_community_detection.py <supernodes.txt>\n")
    sys.exit(1)

communities = {}

fp = open(sys.argv[1])
for line in fp:
    tokens = line.split()
    assert tokens[0] == 'Supernode'

    supernode_id = tokens[1]
    community_id = tokens[4]
    if community_id not in communities:
        communities[community_id] = []

    communities[community_id].append(supernode_id)

for community in communities:
    print(community)
    print('--------------------')
    for supernode in communities[community]:
        print(supernode)
    print('')
