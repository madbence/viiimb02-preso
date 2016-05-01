preso.pdf: preso.tex pseudo-points.csv quasi-points.csv
	xelatex preso.tex

pseudo-points.csv: host-pseudo
	./host-pseudo > pseudo-points.csv

host-pseudo: host-pseudo.c
	nvcc -lcurand -o host-pseudo host-pseudo.c

quasi-points.csv: host-quasi
	./host-quasi > quasi-points.csv

host-quasi: host-quasi.c
	nvcc -lcurand -o host-quasi host-quasi.c
