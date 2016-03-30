all:
	@echo "usage:"
	@echo "  make naive"
	@echo "  make pthread"
	@echo "  make openmp"

clean:
	rm -f assignment1

naive: assignment1.c
	@gcc assignment1.c -std=gnu99 -D_SVID_SOURCE -D_XOPEN_SOURCE=600 -DNAIVE -lm -g -O3 -o assignment1

pthread: assignment1.c
	@gcc assignment1.c -std=gnu99 -D_SVID_SOURCE -D_XOPEN_SOURCE=600 -DPTHREAD -pthread -lm -g -O3 -o assignment1

openmp: assignment1.c
	@gcc assignment1.c -std=gnu99 -D_SVID_SOURCE -D_XOPEN_SOURCE=600 -DOPENMP -fopenmp -lm -g -O3 -o assignment1