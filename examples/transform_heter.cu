extern "C" __global__ void transform_heter
(unsigned int* array, unsigned int first, unsigned int last)
{
  const unsigned int nelems = last - first;
  const unsigned int per_thread = nelems / blockDim.x;
  unsigned int i = first + threadIdx.x * per_thread;
  const unsigned int j = i + per_thread;

  for (; i < j; ++i)
    ++array[i];
}
