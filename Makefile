preso.pdf: preso.tex pseudo-points.csv quasi-points.csv uniform-hist.csv normal-hist.csv
	xelatex preso.tex

pseudo-points.csv: host-pseudo
	./host-pseudo > pseudo-points.csv

host-pseudo: host-pseudo.c
	nvcc -lcurand -o host-pseudo host-pseudo.c

quasi-points.csv: host-quasi
	./host-quasi > quasi-points.csv

host-quasi: host-quasi.c
	nvcc -lcurand -o host-quasi host-quasi.c

uniform-hist.csv: device-uniform
	./device-uniform > uniform-hist.csv

device-uniform: device-uniform.cu
	nvcc -lcurand -lcudart -o device-uniform device-uniform.cu

normal-hist.csv: device-normal
	./device-normal > normal-hist.csv

device-normal: device-normal.cu
	nvcc -lcurand -lcudart -o device-normal device-normal.cu
