extern "C" __global__ void add1
(unsigned int* array, unsigned int nelems)
{
  const unsigned int per_thread = nelems / blockDim.x;
  unsigned int i = threadIdx.x * per_thread;

  unsigned int j = nelems;
  if (threadIdx.x != (blockDim.x - 1)) j = i + per_thread;

  for (; i < j; ++i) ++array[i];
}

extern "C" __global__ void mul2
(unsigned int* array, unsigned int nelems)
{
  const unsigned int per_thread = nelems / blockDim.x;
  unsigned int i = threadIdx.x * per_thread;

  unsigned int j = nelems;
  if (threadIdx.x != (blockDim.x - 1)) j = i + per_thread;

  for (; i < j; ++i) array[i] *= 2;
}
