COMPILER = g++
FLAGS = -Wall -Werror

all: input.h init.h init.o spatial.h spatial.o stringID.h stringID.o main.o
	$(COMPILER) $(FLAGS) init.o spatial.o stringID.o main.o -o main

clean:
	rm -f *.o main

main.o:
	$(COMPILER) -c $(FLAGS) main.cpp

init.o:
	$(COMPILER) -c $(FLAGS) init.cpp

spatial.o:
	$(COMPILER) -c $(FLAGS) spatial.cpp

stringID.o:
	$(COMPILER) -c $(FLAGS) stringID.cpp