hw: main.o render.o
	g++ -lGL -lGLEW -lglfw -I/opt/cuda/include -L/opt/cuda/lib64 -lcudart -lcurand -o hw render.o main.o

main.o: main.cpp
	g++ -c -O3 main.cpp

render.o: render.cu
	nvcc -std=c++11 -c -O3 render.cu

clean:
	rm hw main.o render.o
