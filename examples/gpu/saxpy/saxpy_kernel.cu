
extern "C" __global__ void saxpy_kernel( const float *x, float *y,  const float a,
		const unsigned int first, const unsigned int last )
{
	const unsigned int nelems = last - first;
	const unsigned int per_thread = nelems / blockDim.x;
	unsigned int i = first + threadIdx.x * per_thread;

	unsigned int j = last;
	if (threadIdx.x != (blockDim.x - 1))
		j = i + per_thread;

	for (; i < j; ++i)
		y[i] += a * x[i];
}
