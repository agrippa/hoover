GRAPH500_DIR=$(HOME)/hoover/graph500-graph500-3.0.0
FLAGS=-O2 -g

generate_graph: generate_graph.c
	cc $(FLAGS) generate_graph.c $(GRAPH500_DIR)/generator/graph_generator.c \
		$(GRAPH500_DIR)/generator/utils.c \
		$(GRAPH500_DIR)/generator/splittable_mrg.c \
		-I$(GRAPH500_DIR)/generator -o generate_graph

partition_graph: partition_graph.c
	cc $(FLAGS) partition_graph.c -o partition_graph

clean:
	rm -f generate_graph partition_graph
