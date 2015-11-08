hw: main.o render.o
	g++ -m64 -lGL -lGLEW -lglfw -I/opt/cuda/include -L/opt/cuda/lib64 -lcudart -lcurand -o hw render.o main.o

main.o: main.cpp
	g++ -c main.cpp

render.o: render.cu
	nvcc -std=c++11 -c render.cu

clean:
	rm hw main.o render.o
