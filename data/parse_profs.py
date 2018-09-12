#!/usr/bin/python

import os
import sys

if len(sys.argv) != 2:
    sys.stderr.write('usage: python parse_profs.py <dump-dir>\n')
    sys.exit(1)

dump_dir = sys.argv[1]

prof_files = [os.path.join(dump_dir, f) for f in os.listdir(dump_dir) if f.endswith('.prof') and os.path.isfile(os.path.join(dump_dir, f))]

# PE 101 - timestep 2 - total 14.337000 ms
#   start_time_step 1.313000 ms
#   metadata 0.001000 ms (0.000000 0.000000 0.000000)
#   summary 2.859000 ms (0.450000 1.118000 1.291000)
#   neighbor updates 0.035000 ms
#   coupled values 0.007000 ms
#   coupling 0.003000 ms (0)
#   throttling 0.056000 ms (0 spins)
#   9 / 384 PE neighbors 
#   partition window = , 68 / 64000 partitions active for 87 local vertices
#   aborting? 0 - last step? 0 - remote cache hits=0 misses=109, feature cache hits=0 misses=0 quiets=3, avg # edges=-nan

# Header
print('pe,timestep,total,start_time_step,metadata,metadata_fetch_neighbors,' +
      'metadata_quiet_neighbors,metadata_update_metadata,summary,' +
      'summary_actor_partitions,summary_time_window,summary_update,summary_send_updates,summary_buffer_updates,summary_process_updates,summary_local_creates,summary_remote_creates,summary_local_updates,summary_remote_updates,summary_local_deletes,summary_remote_deletes,neighbor_updates,coupled_values,' +
      'coupling,coupling_spins,throttling,throttling_spins,n_neighbors,' +
      'n_active_partitions,n_total_partitions,n_local_vertices,aborting,' +
      'last_step,remote_cache_hits,remote_cache_misses,quiets')

for prof_file in prof_files:
    prof_fp = open(prof_file, 'r')
    line = prof_fp.readline()
    while len(line) > 0:
        # PE 101 - timestep 2 - total 14.337000 ms
        line = line.strip()
        tokens = line.split()
        pe = int(tokens[1])
        timestep = int(tokens[4])
        total = float(tokens[7])

        #   start_time_step 1.313000 ms
        tokens = prof_fp.readline().strip().split()
        start_time_step = float(tokens[1])

        #   metadata 0.001000 ms (0.000000 0.000000 0.000000)
        tokens = prof_fp.readline().strip().split()
        metadata = float(tokens[1])
        metadata_fetch_neighbors = float(tokens[3][1:])
        metadata_quiet_neighbors = float(tokens[4])
        metadata_update_metadata = float(tokens[5][:-1])

        #   summary 2.859000 ms (0.450000 1.118000 1.291000)
        tokens = prof_fp.readline().strip().split()
        summary = float(tokens[1])
        summary_actor_partitions = float(tokens[3][1:])
        summary_time_window = float(tokens[4])
        summary_update = float(tokens[5])
        summary_send_updates = float(tokens[6])
        summary_buffer_updates = float(tokens[7])
        summary_process_updates = float(tokens[8])
        summary_local_creates = float(tokens[9])
        summary_remote_creates = float(tokens[10])
        summary_local_updates = float(tokens[11])
        summary_remote_updates = float(tokens[12])
        summary_local_deletes = float(tokens[13])
        summary_remote_deletes = float(tokens[14][:-1])

        #   neighbor updates 0.035000 ms
        tokens = prof_fp.readline().strip().split()
        neighbor_updates = float(tokens[2])

        #   coupled values 0.007000 ms
        tokens = prof_fp.readline().strip().split()
        coupled_values = float(tokens[2])

        #   coupling 0.003000 ms (0)
        tokens = prof_fp.readline().strip().split()
        coupling = float(tokens[1])
        coupling_spins = int(tokens[3][1:-1])

        #   throttling 0.056000 ms (0 spins)
        tokens = prof_fp.readline().strip().split()
        throttling = float(tokens[1])
        throttling_spins = int(tokens[3][1:])

        #   9 / 384 PE neighbors 
        tokens = prof_fp.readline().strip().split()
        n_neighbors = int(tokens[0])

        #   partition window = , 68 / 64000 partitions active for 87 local vertices
        line = prof_fp.readline().strip()
        tokens = line.split(',')[1].strip().split()
        n_active_partitions = int(tokens[0])
        n_total_partitions = int(tokens[2])
        n_local_vertices = int(tokens[6])

        #   aborting? 0 - last step? 0 - remote cache hits=0 misses=109, feature cache hits=0 misses=0 quiets=3, avg # edges=-nan
        tokens = prof_fp.readline().strip().split()
        aborting = int(tokens[1])
        last_step = int(tokens[5])
        remote_cache_hits = int(tokens[9].split('=')[1])
        remote_cache_misses = int(tokens[10].split('=')[1][:-1])
        quiets = int(tokens[15][:-1].split('=')[1])

        first = True
        for val in [pe,
                    timestep,
                    total,
                    start_time_step,
                    metadata,
                    metadata_fetch_neighbors,
                    metadata_quiet_neighbors,
                    metadata_update_metadata,
                    summary,
                    summary_actor_partitions,
                    summary_time_window,
                    summary_update,
                    summary_send_updates,
                    summary_buffer_updates,
                    summary_process_updates,
                    summary_local_creates,
                    summary_remote_creates,
                    summary_local_updates,
                    summary_remote_updates,
                    summary_local_deletes,
                    summary_remote_deletes,
                    neighbor_updates,
                    coupled_values,
                    coupling,
                    coupling_spins,
                    throttling,
                    throttling_spins,
                    n_neighbors,
                    n_active_partitions,
                    n_total_partitions,
                    n_local_vertices,
                    aborting,
                    last_step,
                    remote_cache_hits,
                    remote_cache_misses,
                    quiets]:
            if not first:
                sys.stdout.write(',')
            sys.stdout.write(str(val))
            first = False
        sys.stdout.write('\n')

        line = prof_fp.readline()

    prof_fp.close()
