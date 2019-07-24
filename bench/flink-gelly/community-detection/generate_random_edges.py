import random
import os
import sys

if len(sys.argv) != 3:
    sys.stderr.write('usage: python generate_random_edges.py <# records> <# vertices>\n')
    sys.exit(1)

nrecords = int(sys.argv[1])
nvertices = int(sys.argv[2])

i = 0
while i < nrecords:
    print(str(random.randint(0, nvertices-1)) + ' ' + str(random.randint(0, nvertices-1)))
    i += 1
