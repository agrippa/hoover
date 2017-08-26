CFLAGS=-Wall -Werror -g -O0 -Iinc

all: bin/libhoover.so bin/edge_set_test bin/sparse_vec_test bin/init_test

bin/libhoover.so: src/hoover.c src/hvr_avl_tree.c
	oshcc $(CFLAGS) $^ -o $@ -shared -fPIC

bin/init_test: test/init_test.c
	oshcc $(CFLAGS) $^ -o $@ -Lbin -lhoover -lm

bin/edge_set_test: test/edge_set_test.c
	oshcc $(CFLAGS) $^ -o $@ -Lbin -lhoover

bin/sparse_vec_test: test/sparse_vec_test.c
	oshcc $(CFLAGS) $^ -o $@ -Lbin -lhoover

clean:
	rm -f bin/*
