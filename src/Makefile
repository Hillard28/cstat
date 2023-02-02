BLIS_PREFIX = $(HOME)/blis
BLIS_INC    = $(BLIS_PREFIX)/include/haswell/
BLIS_LIB    = $(BLIS_PREFIX)/lib/haswell/libblis.a

OTHER_LIBS  = -fopenmp

CC          = clang
CFLAGS      = -O2 -g -std=c99 -pedantic -I$(BLIS_INC)
LINKER      = $(CC)

OBJS        = test.o gd.o sgd.o utils.o

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

all: $(OBJS)
	$(LINKER) $(OBJS) $(BLIS_LIB) $(OTHER_LIBS) -o test
