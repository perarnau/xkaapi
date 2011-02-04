#include <stdint.h>
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
#define CONFIG_LEAF_COUNT 1024 // stops the recursion
#define CONFIG_ELEM_COUNT (CONFIG_LEAF_COUNT * 1000)
#define CONFIG_RANGE_COUNT 3
#define CONFIG_RANGE_CHECK 1

// addone task
struct TaskAddone : public ka::Task<1>::Signature
<ka::RW<ka::range1d<double_type> > > {};

__global__ void addone(double_type* array, unsigned int size)
{
  const unsigned int per_thread = size / blockDim.x;
  unsigned int i = threadIdx.x * per_thread;

  unsigned int j = size;
  if (threadIdx.x != (blockDim.x - 1)) j = i + per_thread;

  for (; i < j; ++i) ++array[i];
}

template<> struct TaskBodyCPU<TaskAddone>
{
  void operator() (ka::range1d_rw<double_type> range)
  {
    const size_t range_size = range.size();
    for (size_t i = 0; i < range_size; ++i)
      range[i] += 1;
  }
};

template<> struct TaskBodyGPU<TaskAddone>
{
  void operator()(ka::gpuStream stream, ka::range1d_rw<double_type> range)
  {
    printf(">>> kudaAddone %lx, %lx\n", (uintptr_t)range.begin(), range.size());
    const CUstream custream = (CUstream)stream.stream;
    addone<<<1, 256, 0, custream>>>(range.begin(), range.size());
    printf("<<< kudaAddone %lx, %lx\n", (uintptr_t)range.begin(), range.size());
  }
};

// init task. needed because of a bug in the runtime.
struct TaskInit : public ka::Task<1>::Signature
<ka::W<ka::range1d<double_type> > > {};

template<> struct TaskBodyCPU<TaskInit>
{
  void operator() (ka::range1d_w<double_type> range)
  {
    const size_t range_size = range.size();
    for (size_t i = 0; i < range_size; ++i)
      range[i] = 0;
  }
};

// fetch memory back to original cpu
struct TaskFetch : public ka::Task<2>::Signature
<uintptr_t, ka::R<ka::range1d<double_type> > > {};

template<> struct TaskBodyCPU<TaskFetch>
{
  void operator()(uintptr_t fubar, ka::range1d_r<double_type> range)
  {
    double_type* const addr = (double_type*)fubar;
    for (size_t i = 0; i < range.size(); ++i)
      addr[i] = range[i];
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
      threadgroup.force_archtype(1, KAAPI_PROC_TYPE_CUDA);

      for (size_t count = 0; count < CONFIG_RANGE_COUNT; ++count)
      {
	double_type* const array = (double_type*)kaapi_mem_alloc_host(total_size);

	memset(array, 0, total_size);
	ka::range1d<double_type> range(array, array + CONFIG_ELEM_COUNT);
	threadgroup.Spawn<TaskInit>(ka::SetPartition(0))(range);
	threadgroup.Spawn<TaskAddone>(ka::SetPartition(1))(range);
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
	size_t i; for (i = 0; i < CONFIG_ELEM_COUNT; ++i)
	  if (array[i] != 1.f) break;
	if (i != CONFIG_ELEM_COUNT)
	  printf("invalid @%lx,%u::%u == %u\n",
		 (uintptr_t)array, count, i, array[i]);
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
