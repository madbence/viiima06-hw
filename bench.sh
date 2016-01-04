#!/bin/bash

rm cuda.csv
rm no-cuda.csv
rm *.ppm *.png
TIMEFORMAT='%E'

for i in 1 2 5 10 20 50 100 200 500; do
  echo $i
  rm render.o
  make hw-ppm BASE_ITER=$i MODE=NOUSE_CUDA
  (time (./hw-ppm > render-no-cuda-$i.ppm; )) 2>tmp
  echo "($i,$(cat tmp | tr -d '\n'))" >>no-cuda.csv
  convert render-no-cuda-$i.ppm render-no-cuda-$i.png
done

for i in 1 2 5 10 20 50 100 200 500; do
  echo $i
  rm render.o
  make hw-ppm BASE_ITER=$i MODE=USE_CUDA
  (time (./hw-ppm > render-cuda-$i.ppm; )) 2>tmp
  echo "($i,$(cat tmp | tr -d '\n'))" >>cuda.csv
  convert render-cuda-$i.ppm render-cuda-$i.png
done
