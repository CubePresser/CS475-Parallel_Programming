all: simd.p4.o arraymult

clean: 
	rm arraymult simd.p4.o

simd.p4.o:	simd.p4.h simd.p4.cpp
		g++ -c simd.p4.cpp -o simd.p4.o

arraymult:	arraymult.cpp simd.p4.o
		g++ -DARRAYSIZE=$(ARRAYSIZE) -o arraymult arraymult.cpp simd.p4.o -lm -fopenmp
