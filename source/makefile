#makefile for the library

CC = gcc
#LDFLAGS = -lm
OBJFILES = mcmcsim.o

all: libmcmcsim.a

libmcmcsim.a:
	$(CC) -c -fPIC -o mcmcsim.o mcmcsim.c
	ar rcs libmcmcsim.a $(OBJFILES)

clean:
	rm -f $(OBJFILES) libmcmcsim.a *~
