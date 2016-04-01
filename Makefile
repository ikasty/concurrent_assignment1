all:
	@echo "usage:"
	@echo "  make naive [options]      - use only 1 thread"
	@echo "  make pthread [options]    - use Pthread"
	@echo "  make openmp [options]     - use OpenMP"
	@echo ""
	@echo "options:"
	@echo "  DEBUG=1        - set debug mode"
	@echo "  NORAND=1       - not using random value"

clean:
	rm -f assignment1

ifeq ($(DEBUG), 1)
OPTION += -DDEBUG_ENABLED
endif
ifeq ($(NORAND), 1)
OPTION += -DNORAND
endif

naive: assignment1.c
	@gcc assignment1.c -std=gnu99 -D_SVID_SOURCE -D_XOPEN_SOURCE=600 $(OPTION) -DNAIVE -lm -g -O3 -o assignment1

pthread: assignment1.c
	@gcc assignment1.c -std=gnu99 -D_SVID_SOURCE -D_XOPEN_SOURCE=600 $(OPTION) -DPTHREAD -pthread -lm -g -O3 -o assignment1

openmp: assignment1.c
	@gcc assignment1.c -std=gnu99 -D_SVID_SOURCE -D_XOPEN_SOURCE=600 $(OPTION) -DOPENMP -fopenmp -lm -g -O3 -o assignment1