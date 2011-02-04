#include <stdint.h>
#include <string.h>
#include <sys/types.h>
#include <cuda.h>
#include "kaapi++"

// missing decls
typedef uintptr_t kaapi_mem_addr_t;
extern "C" void kaapi_mem_delete_host_mappings(kaapi_mem_addr_t, size_t);
extern "C" void* kaapi_mem_alloc_host(size_t);
extern "C" void kaapi_mem_free_host(void*);
extern "C" void* kaapi_mem_find_host_addr(kaapi_mem_addr_t);

// typedefed to float since gtx280 has no double
typedef unsigned int double_type;

// static configuration
#define CONFIG_ITER_COUNT 1
#define CONFIG_ROW_COUNT 8
#define CONFIG_COL_COUNT 32
#define CONFIG_RANGE_COUNT 1
#define CONFIG_RANGE_CHECK 1

// task signature
struct TaskAddone : public ka::Task<1>::Signature
<ka::RW<ka::range2d<double_type> > > {};

// cuda kernel
__global__ void addone
(double_type* array, unsigned int rows, unsigned int cols)
{
  const unsigned int col_per_thread = cols / blockDim.x;
  unsigned int col = threadIdx.x * col_per_thread;
  unsigned int last_col = cols;
  if (threadIdx.x != (blockDim.x - 1)) last_col = col + col_per_thread;

  const unsigned int row_per_thread = rows / blockDim.y;
  unsigned int row = threadIdx.y * row_per_thread;
  unsigned int last_row = rows;
  if (threadIdx.y != (blockDim.y - 1)) last_row = row + row_per_thread;

  __syncthreads();

  array[0] = 42;

  for (; row < last_row; ++row)
    for (; col < last_col; ++col)
      array[0] = 42;
      // ++array[row * cols + col];
}


// lazyness symphony

template<typename barfu>
inline static size_t rows(const barfu& o)
{ return o.dim(0); }

template<typename barfu>
inline static size_t cols(const barfu& o)
{ return o.dim(1); }


// cpu implementation

template<> struct TaskBodyCPU<TaskAddone>
{
  void operator()(ka::range2d_rw<double_type> range)
  {
    for (size_t row = 0; row < rows(range); ++row)
      for (size_t col = 0; col < cols(range); ++col)
	range(row, col) += 1;
  }

};

// gpu implementation
template<> struct TaskBodyGPU<TaskAddone>
{
  void operator()(ka::gpuStream stream, ka::range2d_rw<double_type> range)
  {
    printf(">>> kudaAddone %lx, %u %u\n",
	   (uintptr_t)&range(0, 0), rows(range), cols(range));

    const CUstream custream = (CUstream)stream.stream;

    static const dim3 fubar(256, 256, 1);
    addone<<<1, fubar, 0, custream>>>
      (&range(0, 0), rows(range), cols(range));
  }
};


// fetch memory back to original cpu
struct TaskFetch : public ka::Task<2>::Signature
<uintptr_t, ka::R<ka::range2d<double_type> > > {};

template<> struct TaskBodyCPU<TaskFetch>
{
  void operator()(uintptr_t fubar, ka::range2d_r<double_type> range)
  {
    double_type* const addr = (double_type*)fubar;

    printf(">>> TaskFetch %lx <- %lx\n", addr, (uintptr_t)&range(0, 0));

#if 0 // uncomment if valid
    const size_t size = rows(range) * cols(range) * sizeof(double_type);
    memcpy(addr, range.begin(), size);
#else
    for (size_t row = 0; row < rows(range); ++row)
      for (size_t col = 0; col < cols(range); ++col)
	addr[row * cols(range) + col] = range(row, col);
#endif
  }
};


/* Main of the program
*/
struct doit {
  void operator()(int argc, char** argv )
  {
    double_type* arrays[CONFIG_RANGE_COUNT];
    const size_t total_size =
      CONFIG_ROW_COUNT * CONFIG_COL_COUNT * sizeof(double_type);

    double t0,t1, sum = 0.f;

    for (size_t iter = 0; iter < CONFIG_ITER_COUNT; ++iter)
    {
      t0 = kaapi_get_elapsedns();

      // prepare partitions
      ka::ThreadGroup threadgroup(2);

      threadgroup.begin_partition();
      threadgroup.force_archtype(1, KAAPI_PROC_TYPE_CUDA);

      for (size_t count = 0; count < CONFIG_RANGE_COUNT; ++count)
      {
	double_type* const array = (double_type*)kaapi_mem_alloc_host(total_size);

	memset(array, 0, total_size);
	ka::range2d<double_type> range
	  (array, CONFIG_ROW_COUNT, CONFIG_COL_COUNT, CONFIG_COL_COUNT);
	threadgroup.Spawn<TaskAddone>(ka::SetPartition(1))(range);
	printf("spawn(%u, %lx)\n", count, (uintptr_t)array);
	threadgroup.Spawn<TaskFetch>(ka::SetPartition(0))((uintptr_t)array, range);
	arrays[count] = array;
      }

      threadgroup.end_partition();

      threadgroup.execute();

      t1 = kaapi_get_elapsedns();
      sum += (t1-t0)/1000; // us

      // check it
#if CONFIG_RANGE_CHECK
      printf(">>> checking range\n");
      for (size_t count = 0; count < CONFIG_RANGE_COUNT; ++count)
      {
	double_type* const array = arrays[count];
	for (size_t row = 0; row < CONFIG_ROW_COUNT; ++row)
	  for (size_t col = 0; col < CONFIG_COL_COUNT; ++col)
	  {
	    const double_type value = array[row * CONFIG_COL_COUNT + col];
	    if (value != 1)
	    {
	      printf("invalid @%u(%u,%u)::%u == %u\n", count, row, col, value);
	      row = CONFIG_ROW_COUNT - 1;
	      break;
	    }
	  }
      }
      printf("<<< checking range\n");
#endif

      for (size_t count = 0; count < CONFIG_RANGE_COUNT; ++count)
      {
	kaapi_mem_delete_host_mappings
	  ((kaapi_mem_addr_t)arrays[count], total_size);
	kaapi_mem_free_host(arrays[count]);
      }
    }

    printf("time: %lf (us)\n", sum / CONFIG_ITER_COUNT);
  }
};


/*
*/
int main( int argc, char** argv ) 
{
  try {
    ka::Community com = ka::System::join_community( argc, argv );

    ka::SpawnMain<doit>()(argc, argv); 

    com.leave();

    ka::System::terminate();
  }
  catch (const ka::Exception& E) {
    ka::logfile() << "Catch : "; E.print(std::cout); std::cout << std::endl;
  }
  catch (...) {
    ka::logfile() << "Catch unknown exception: " << std::endl;
  }
  return 0;    
}
