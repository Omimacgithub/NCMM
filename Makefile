CC = nvcc

#Debbuging symbols to analize with gdb
CFLAGS = -Xcompiler -Wall -Xcompiler -Wextra -Xcudafe --display_error_number -lcublas -lineinfo 

MM: MM.cu
	nvcc $(DFLAG) $(CFLAGS) $< -o $@
MMo: MMo.cu
	nvcc $(DFLAG) $(CFLAGS) $< -o $@
MMo1: MMo1.cu
	nvcc $(DFLAG) $(CFLAGS) $< -o $@
MMo2: MMo2.cu
	nvcc $(DFLAG) $(CFLAGS) $< -o $@
