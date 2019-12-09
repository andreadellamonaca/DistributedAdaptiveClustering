OBJS	= main.o adaptive_clustering.o graph.o error.o
SOURCE	= main.cpp adaptive_clustering.cpp graph.cpp error.cpp
HEADER	= adaptive_clustering.h graph.h error.h
OUT	= p2pAdaptiveClustering.out
CC	= g++
FLAGS	= -g -c -Wall
CFLAGS	= -I /usr/local/include/igraph
LFLAGS	= -L /usr/local/lib
LIBS	= -ligraph -larmadillo 

all: $(OBJS)
	$(CC) -g $(OBJS) -o $(OUT) $(CFLAGS) $(LFLAGS) $(LIBS)


main.o: main.cpp
	$(CC) $(FLAGS) main.cpp -std=c++14

adaptive_clustering.o: adaptive_clustering.cpp
	$(CC) $(FLAGS) adaptive_clustering.cpp -std=c++14

graph.o: graph.cpp
	$(CC) $(FLAGS) graph.cpp -std=c++14

error.o: error.cpp
	$(CC) $(FLAGS) error.cpp -std=c++14


clean:
	rm -f $(OBJS) $(OUT)

