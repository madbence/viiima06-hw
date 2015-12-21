hw-opengl: main.o render.o
	g++ -lGL -lGLEW -lglfw -I/opt/cuda/include -L/opt/cuda/lib64 -lcudart -lcurand -o hw-opengl render.o main.o

hw-ppm: main2.o render.o
	g++ -I/opt/cuda/include -L/opt/cuda/lib64 -lcudart -lcurand -o hw-ppm render.o main2.o

main.o: main.cpp
	g++ -c -O3 main.cpp

main2.o: main2.cpp
	g++ -c -O3 main2.cpp

render.o: render.cu
	nvcc -std=c++11 -c -O3 render.cu -DUSE_CUDA

clean:
	rm hw main.o render.o
