# CUDA Prefix Sum
This repository provides a CUDA prefix sum implementation for arbitrary **even** size array.

## Usage
Include cuda_prefix_sum function in your code	

    void cuda_prefix_sum(int *input_array,int *output_array,int count)
   
   The args of the function are 
 - int pointer for elements to be summed
 - int pointer to store prefix sum result. This pointer needs to be properly allocated
 - count is the size of the array. 

Make sure to update the following constants based on your GPU
 - MAX_THREADS_PER_BLOCK
 - MAX_NUM_GPU_BLOCKS

## Description
cuda_prefix_sum function can take any arbitrary **even** size array. If you have odd array just add one dummy element to the end.

The function passes tests with arrays of size up to 1024*1024*128. Bigger arrays were tested but started to get out of memory error. So as long as you have enough memory the function works fine.

## Testing
To test the code, simply make the executable using the makefile. Then run ./main.o command.
The code includes eight tests

## Contribution

One thing I wish to change in this code is the first loop

    for(int i=0,block=0;i<count;i+=max_elements_per_block,block++)
  
I need to read more about GPU shared memory before doing so. The current function implementation is sufficient for my usage, so I didn't improve it. 

If someone have a better fix for this issue, you are welcomed :)

