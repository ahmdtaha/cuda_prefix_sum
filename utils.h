#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
    std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
    exit(1);
  }
}


void print_array(int* in_array,int count)
{
	int sum = 0;
	for(int i=0;i<count;i++){
		printf("%d,",in_array[i]);
		sum += in_array[i];
		if((i+1) % 8 == 0)
		{
			printf("=%d\n",sum);
			sum =0;
		}
	}
	printf("=%d\n",sum);
}

bool is_power_two(ulong x)
{
    return (x & (x - 1)) == 0;
}

int next_power_two(int n)
{
	n--;
	n |= n >> 1;   // Divide by 2^k for consecutive doublings of k up to 32,
	n |= n >> 2;   // and then or the results.
	n |= n >> 4;
	n |= n >> 8;
	n |= n >> 16;
	n++;
	return n;      
}