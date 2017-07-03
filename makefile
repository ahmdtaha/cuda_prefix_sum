NVCC=/usr/local/cuda/bin/nvcc
#NVCC=nvcc

###################################
# These are the default install   #
# locations on most linux distros #
###################################

#OPENCV_LIBPATH=/usr/lib
#OPENCV_INCLUDEPATH=/usr/include

###################################################
# On Macs the default install locations are below #
###################################################

OPENCV_LIBPATH=/usr/local/lib
OPENCV_INCLUDEPATH=/usr/local/include/opencv 
OPENCV_INCLUDEPATH2=/usr/local/include

OPENCV_LIBS=$(shell pkg-config --libs opencv) ## One way of doing it
OPENCV_LIBS=`pkg-config --libs opencv` ## One way of doing it

CUDA_INCLUDEPATH=/usr/local/cuda/include
# CUDA_INCLUDEPATH=/usr/local/cuda/lib64/include
# CUDA_INCLUDEPATH=/usr/local/cuda/include
# CUDA_INCLUDEPATH=/Developer/NVIDIA/CUDA-5.0/include

######################################################
# On Macs the default install locations are below    #
# ####################################################

#CUDA_INCLUDEPATH=/usr/local/cuda/include
#CUDA_LIBPATH=/usr/local/cuda/lib
CUDA_LIBPATH=/usr/local/cuda/lib64

NVCC_OPTS=-O3 -arch=sm_20 -Xcompiler -Wall -Xcompiler -Wextra -m64 -Wno-deprecated-gpu-targets

GCC_OPTS=-O3 -Wall -Wextra -m64


main.o: main.cu utils.h
	$(NVCC) -o main.o main.cu $(NVCC_OPTS) -I $(CUDA_INCLUDEPATH)


clean:
	rm -f *.o *.png hw
