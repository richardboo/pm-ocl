OBJS = main.o apr_ocl_utils.o
CC = g++
#DEBUG = -g
PROFILE = -D ENABLE_PROFILER
CFLAGS = -c $(DEBUG) $(PROFILE) -std=c++11
LFLAGS = $(DEBUG) -framework OpenCL

pm: $(OBJS)
	$(CC) $(LFLAGS) $(OBJS) -o pm

main.o: main.cpp apr_ocl_utils.cpp apr_ocl_utils.hpp
	$(CC) $(CFLAGS) main.cpp

apr_ocl_utils.o: apr_ocl_utils.cpp apr_ocl_utils.hpp
	$(CC) $(CFLAGS) apr_ocl_utils.cpp

clean:
	\rm *.o *~ pm