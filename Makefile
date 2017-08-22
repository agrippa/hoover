all: bin/libhoover.so bin/init_test

bin/libhoover.so: src/hoover.c
	oshcc $^ -o $@ -shared -fPIC -g -O0 -Iinc

bin/init_test: test/init_test.c
	oshcc $^ -o $@ -g -O0 -Iinc -Lbin -lhoover

clean:
	rm -f bin/*
