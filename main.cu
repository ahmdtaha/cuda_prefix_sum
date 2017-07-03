#include <stdio.h>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <math.h>

using namespace std;

#include "utils.h"


// This constant defines the maximum threads you can run on a single GPU block.
// This value depends on your GPU. It affects the number of elements 
#define MAX_THREADS_PER_BLOCK 1024
#define MAX_NUM_GPU_BLOCKS 1024

__global__  void increment_prefix_sum(int* g_odata,int* d_block_sums,int n,int block_shift=0)
{
	int thid = blockIdx.x * blockDim.x + threadIdx.x;
	if(2*thid > n)
		return;
	g_odata[2*thid] += d_block_sums[blockIdx.x+block_shift];
	g_odata[2*thid+1] += d_block_sums[blockIdx.x+block_shift];	
	
	
}

/**
This function computes the prefix sum for a single block that fits in a single streaming multiprocessor.
In such case, set the save_sum bool flag to false and provide the total_sum ptr to NULL

If you want to use it compute prefix sum for arbitrary size array, you need to store the total block sum in block sums array
In such case, set the save_sum bool flag to true and provide a valid ptr to save the total sum
**/
__global__  void cuda_block_prefix(int* g_odata, int* g_idata, int n,bool save_sum,int *total_sum)
{
	extern __shared__  int temp[];
	int thid = threadIdx.x; 
	int offset = 1;

	if(2*thid < n)
		temp[2*thid] = g_idata[2*thid];
	else
		temp[2*thid] = 0;
	

	if(2*thid + 1 < n)
		temp[2*thid+1] = g_idata[2*thid+1]; 
	else
		temp[2*thid+1] = 0; 
	

	for (int d = n>>1; d > 0; d >>= 1){
		__syncthreads(); 
		if (thid < d){
			int ai = offset*(2*thid+1)-1;
			int bi = offset*(2*thid+2)-1;
			temp[bi] += temp[ai];  
		}    
		offset *= 2; 
	}

 
	if (thid == 0) 
	{
		if(save_sum)
			*total_sum = temp[n - 1];
			

		temp[n - 1] = 0; 
	}

	for (int d = 1; d < n; d *= 2){
		offset >>= 1;
		__syncthreads(); 
		if (thid < d){
			int ai = offset*(2*thid+1)-1;
			int bi = offset*(2*thid+2)-1;
			int t   = temp[ai];
			temp[ai]  = temp[bi];
			temp[bi]  += t;   
		} 
	}

	__syncthreads();
	g_odata[2*thid] = temp[2*thid];
	g_odata[2*thid+1] = temp[2*thid+1];
	
}

/*
This functions computes the prefix sum of arbitrary *even* size array. This code supports up to size 2097152 (1024*2*1024)
If your array size is odd, you can addend one dummy element and use this method
Make sure to configure MAX_THREADS_PER_BLOCK correctly to avoid errors and maximize your GPU utilization

If big size array support is required, the 
*/
void cuda_prefix_sum(int *input_array,int *output_array,int count)
{	
	int max_elements_per_block = 2 * MAX_THREADS_PER_BLOCK;
	int num_blocks = ceil((float)count / max_elements_per_block);
	num_blocks = next_power_two(num_blocks);

	

	int *d_elements,*d_cuda_prefix_sum,*d_sum_blocks,*d_blocks_inc;
	checkCudaErrors(cudaMalloc(&d_elements,  sizeof(int) * count));
  	checkCudaErrors(cudaMalloc(&d_cuda_prefix_sum,  sizeof(int) * count));
  	checkCudaErrors(cudaMalloc(&d_sum_blocks,  sizeof(int) * num_blocks));
  	
  	
	cudaMemcpy(d_elements, input_array, sizeof(int) * count, cudaMemcpyHostToDevice);
	checkCudaErrors(cudaMemset(d_cuda_prefix_sum,0,sizeof(int) * count));
	checkCudaErrors(cudaMalloc(&d_blocks_inc,  sizeof(int) * num_blocks));
	checkCudaErrors(cudaMemset(d_blocks_inc,0,sizeof(int) * num_blocks));
			

	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	int unprocessed = count;
	for(int i=0,block=0;i<count;i+=max_elements_per_block,block++)
	{
			
		if(unprocessed >= max_elements_per_block){
			unprocessed -= max_elements_per_block;
			cuda_block_prefix<<<1,MAX_THREADS_PER_BLOCK,max_elements_per_block*sizeof(int)>>>(&d_cuda_prefix_sum[i],&d_elements[i],max_elements_per_block,true,&d_sum_blocks[block]);
			cudaMemcpy(&output_array[i], &d_cuda_prefix_sum[i], sizeof(int) * max_elements_per_block, cudaMemcpyDeviceToHost);
			cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
			
		}
		else{
			int last_block_size = next_power_two(unprocessed);
			cuda_block_prefix<<<1,MAX_THREADS_PER_BLOCK,max_elements_per_block*sizeof(int)>>>(&d_cuda_prefix_sum[i],&d_elements[i],last_block_size,true,&d_sum_blocks[block]);
			cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
			cudaMemcpy(&output_array[i], &d_cuda_prefix_sum[i], sizeof(int) * unprocessed, cudaMemcpyDeviceToHost);
			cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
		}

	}

	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	int  threads_for_blocks= next_power_two(num_blocks);

	if(threads_for_blocks >1)
	{
		
		if(num_blocks <= MAX_THREADS_PER_BLOCK * 2)
		{
			cuda_block_prefix<<<1,threads_for_blocks/2,threads_for_blocks*sizeof(int)>>>(d_blocks_inc,d_sum_blocks,num_blocks,false,NULL);
		}
		else
		{
			int* blocks_sum_input = new int[num_blocks];
			int *block_inc_output = new int[num_blocks];
			cudaMemcpy(blocks_sum_input, d_sum_blocks, sizeof(int) * num_blocks, cudaMemcpyDeviceToHost);
			cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
			cuda_prefix_sum(blocks_sum_input,block_inc_output,num_blocks);
			cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
			cudaMemcpy(d_blocks_inc, block_inc_output, sizeof(int) * num_blocks, cudaMemcpyHostToDevice);
			cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
			free(blocks_sum_input);
			free(block_inc_output);
		}
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
		int num_gpu_blocks = ceil(count/(2.0*MAX_THREADS_PER_BLOCK));
		
		if(num_gpu_blocks <= MAX_NUM_GPU_BLOCKS)
		{
			increment_prefix_sum<<<ceil(count/(2.0*MAX_THREADS_PER_BLOCK)),MAX_THREADS_PER_BLOCK>>>(&d_cuda_prefix_sum[0],d_blocks_inc,count);
		}
		else
		{
			int count_processed = 0;
			for(int i=0;i<num_gpu_blocks;i+=MAX_NUM_GPU_BLOCKS)
			{
				count_processed += MAX_NUM_GPU_BLOCKS * MAX_THREADS_PER_BLOCK * 2;
				increment_prefix_sum<<<MAX_NUM_GPU_BLOCKS,MAX_THREADS_PER_BLOCK>>>(&d_cuda_prefix_sum[2*i*MAX_THREADS_PER_BLOCK],d_blocks_inc,count,i);
			}
		}
		cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	}	
	
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	cudaMemcpy(output_array, d_cuda_prefix_sum, sizeof(int) * count, cudaMemcpyDeviceToHost);
	cudaFree(d_elements);
	cudaFree(d_cuda_prefix_sum);
	cudaFree(d_sum_blocks);
	cudaFree(d_blocks_inc);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}

void test_array_size(int count,int test_no)
{
	bool verbose = false;
	int* elements = new int[count];
	int* seq_prefix_sum = new int[count];
	int* gpu_prefix_sum = new int[count];

	
	//srand (0); // For debugging purpose
	srand (time(NULL));


	for(int i=0;i<count;i++)
		elements[i] = rand() % 10 + 1;     //v2 = rand() % 100 + 1;     //
	
	if(verbose)
	{
		printf("Original Elements\n");
		print_array(elements,count);
	}
	
	seq_prefix_sum[0] = 0;
	for(int j = 1; j < count; ++j)
		seq_prefix_sum[j] = elements[j-1] + seq_prefix_sum[j-1];

	if(verbose)
	{
		printf("Sequential Prefix sum\n");
		print_array(seq_prefix_sum,count);	
	}
	

	cuda_prefix_sum(elements,gpu_prefix_sum,count);
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	if(verbose)
		print_array(gpu_prefix_sum,count);

	bool match=true;
	for(int i=0;i<count;i++)
		if(gpu_prefix_sum[i] != seq_prefix_sum[i])
		{
			match = false;
			break;
		}
	if(match)
		cout << "Test " << test_no <<" Passed\n";
	else
		cout << "Test " << test_no <<" Failed\n";

	free(elements);
	free(seq_prefix_sum);
	free(gpu_prefix_sum);
	
}
void test1()
{
	int count = 4*4*3;
	test_array_size(count,1);
}

void test2()
{
	int count = 1024*2;
	test_array_size(count,2);
}

void test3()
{
	int count = 1024*100;
	test_array_size(count,3);
}
void test4()
{
	int count = 1024*1024;
	test_array_size(count,4);
}

void test5()
{
	int count = 1024*1024*2;
	test_array_size(count,5);
}


void test6()
{
	int count = 1024*1024*2*2;
	test_array_size(count,6);
}

void test7()
{
	int count = 1024*1024*4;
	test_array_size(count,7);
}

void test8()
{
	int count = 1024*1024*128;
	test_array_size(count,8);
}


int main()
{
	test1();
	//return;
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	test2();
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	test3();
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	test4();
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	test5();
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	test6();
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	test7();
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
	test8();
	
	return 0;
}