OBJS = main.o apr_ocl_utils.o
CC = g++
CFLAGS = -c -g -std=c++11 $(DEBUG)
LFLAGS = -g -framework OpenCL

pm: $(OBJS)
	$(CC) $(LFLAGS) $(OBJS) -o pm

main.o: main.cpp apr_ocl_utils.cpp apr_ocl_utils.hpp
	$(CC) $(CFLAGS) main.cpp

apr_ocl_utils.o: apr_ocl_utils.cpp apr_ocl_utils.hpp
	$(CC) $(CFLAGS) apr_ocl_utils.cpp

clean:
	\rm *.o *~ pm