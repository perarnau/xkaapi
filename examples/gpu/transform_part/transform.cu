#include <stdint.h>
#include <sys/types.h>
#include <cuda.h>
#include "kaapi++"

// missing decls
typedef uintptr_t kaapi_mem_addr_t;
extern "C" void kaapi_mem_delete_host_mappings(kaapi_mem_addr_t, size_t);
extern "C" void* kaapi_mem_alloc_host(size_t);
extern "C" void kaapi_mem_free_host(void*);

// typedefed to float since gtx280 has no double
typedef float double_type;

// static configuration
#define CONFIG_ITER_COUNT 50
#define CONFIG_LEAF_COUNT 1024 // stops the recursion
#define CONFIG_ELEM_COUNT (CONFIG_LEAF_COUNT * 1000)
#define CONFIG_RANGE_COUNT 3
#define CONFIG_RANGE_CHECK 0

// task signature
struct TaskAddone : public ka::Task<1>::Signature
<ka::RW<ka::range1d<double_type> > > {};

// cuda kernel
__global__ void addone(double_type* array, unsigned int size)
{
#if 0
  const unsigned int per_thread = size / blockDim.x;
  unsigned int i = threadIdx.x * per_thread;

  unsigned int j = size;
  if (threadIdx.x != (blockDim.x - 1)) j = i + per_thread;

  for (; i < j; ++i) ++array[i];
#endif
}

// cpu implementation
template<> struct TaskBodyCPU<TaskAddone>
{
  void operator() (ka::range1d_rw<double_type> range)
  {
    const size_t range_size = range.size();

#if 0
    // reached the leaf size
    if (range_size <= CONFIG_LEAF_COUNT)
    {
#endif
      for (size_t i = 0; i < range_size; ++i)
	range[i] += 1;
      return ;
#if 0
    }

    // recurse by spawning 2 tasks with 1/2 range each

    double_type* const lend = range.begin() + range_size / 2;

    ka::range1d<double_type> l(range.begin(), lend);
    ka::range1d<double_type> r(lend, range_size - l.size());

    ka::Spawn<TaskAddone>()(l);
    ka::Spawn<TaskAddone>()(r);
#endif
  }

};

// gpu implementation
template<> struct TaskBodyGPU<TaskAddone>
{
  void operator()(ka::gpuStream stream, ka::range1d_rw<double_type> range)
  {
    // we are the big one, can handle the work alone. dont recurse.
    const CUstream custream = (CUstream)stream.stream;
    addone<<<1, 256, 0, custream>>>(range.begin(), range.size());
  }
};

/* Main of the program
*/
struct doit {
  void operator()(int argc, char** argv )
  {
    double_type* arrays[CONFIG_RANGE_COUNT];
    const size_t total_size = CONFIG_ELEM_COUNT * sizeof(double_type);

    double t0,t1, sum = 0.f;

    for (size_t iter = 0; iter < CONFIG_ITER_COUNT; ++iter)
    {
      t0 = kaapi_get_elapsedns();

      // prepare partitions
      ka::ThreadGroup threadgroup(2);

      threadgroup.begin_partition();

      for (size_t count = 0; count < CONFIG_RANGE_COUNT; ++count)
      {
	double_type* const array = (double_type*)
	  kaapi_mem_alloc_host(total_size);
	memset(array, 0, total_size);
	ka::range1d<double_type> range(array, array + CONFIG_ELEM_COUNT);
	threadgroup.Spawn<TaskAddone>(ka::SetPartition(1))(range);
	arrays[count] = array;
      }

      threadgroup.end_partition();

      threadgroup.execute();

      t1 = kaapi_get_elapsedns();
      sum += (t1-t0)/1000; // us

      // check it
#if CONFIG_RANGE_CHECK
      for (size_t count = 0; count < CONFIG_RANGE_COUNT; ++count)
      {
	double_type* const array = arrays[count];
	size_t i; for (i = 0; i < CONFIG_ELEM_COUNT; ++i)
	  if (array[i] != 1.f) break;
	if (i != CONFIG_ELEM_COUNT)
	  printf("invalid @%u\n", i);
      }
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
