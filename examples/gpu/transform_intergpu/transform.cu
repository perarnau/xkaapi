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
extern "C" unsigned int kaapi_cuda_get_kasid_user(size_t);
extern "C" size_t kaapi_cuda_get_proc_count(void);
extern "C" unsigned int kaapi_cuda_get_first_kid(void);

// typedefed to float since gtx280 has no double
typedef unsigned int double_type;

// static configuration
#define CONFIG_ITER_COUNT 1
#define CONFIG_ROW_COUNT 16
#define CONFIG_COL_COUNT 16
#define CONFIG_RANGE_CHECK 1


template<typename barfu>
inline static size_t rows(const barfu& o)
{ return o.dim(0); }

template<typename barfu>
inline static size_t cols(const barfu& o)
{ return o.dim(1); }


// addone task

struct TaskAddone : public ka::Task<1>::Signature
<ka::RW<ka::range2d<double_type> > > {};

__global__ void addone(double_type* array)
{
  ++array[threadIdx.y * blockDim.x + threadIdx.x];
}

template<> struct TaskBodyCPU<TaskAddone>
{
  void operator()(ka::range2d_rw<double_type> range)
  {
    for (size_t row = 0; row < rows(range); ++row)
      for (size_t col = 0; col < cols(range); ++col)
	range(row, col) += 1;
  }

};

template<> struct TaskBodyGPU<TaskAddone>
{
  void operator()(ka::gpuStream stream, ka::range2d_rw<double_type> range)
  {
    printf(">>> kudaAddone %lx\n", (uintptr_t)&range(0, 0));

    const CUstream custream = (CUstream)stream.stream;

    static const dim3 fubar(16, 16);
    addone<<<1, fubar, 0, custream>>>(&range(0, 0));
  }
};


// first[i] += second[i] task

struct TaskAddranges : public ka::Task<2>::Signature
<ka::RW<ka::range2d<double_type> > , ka::R<ka::range2d<double_type> > > {};

__global__ void addranges(double_type* first, double_type* second)
{
  // first the inout
  const size_t i = threadIdx.y * blockDim.x + threadIdx.x;
  first[i] += second[i];
}

template<> struct TaskBodyCPU<TaskAddranges>
{
  void operator()
  (ka::range2d_rw<double_type> first, ka::range2d_r<double_type> second)
  {
    // assume dim(first) == dim(second)
    for (size_t row = 0; row < rows(first); ++row)
      for (size_t col = 0; col < cols(first); ++col)
	first(row, col) += second(row, col);
  }

};

template<> struct TaskBodyGPU<TaskAddranges>
{
  void operator()
  (
   ka::gpuStream stream,
   ka::range2d_rw<double_type> first,
   ka::range2d_r<double_type> second
  )
  {
    printf(">>> kudaAddranges %lx %lx\n",
	   (uintptr_t)&first(0, 0),
	   (uintptr_t)&second(0, 0));

    const CUstream custream = (CUstream)stream.stream;

    static const dim3 fubar(16, 16);
    addranges<<<1, fubar, 0, custream>>>(&first(0, 0), &second(0, 0));
  }
};



// init task. needed because of a bug in the runtime.
struct TaskInit : public ka::Task<1>::Signature
<ka::W<ka::range2d<double_type> > > {};

template<> struct TaskBodyCPU<TaskInit>
{
  void operator() (ka::range2d_w<double_type> range)
  {
    printf(">>> TaskInit %lx\n", (uintptr_t)&range(0, 0));

    for (size_t row = 0; row < rows(range); ++row)
      for (size_t col = 0; col < cols(range); ++col)
	range(row, col) = 0;
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
    memcpy(addr, (void*)&range(0, 0), size);
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
    const size_t gpu_count = kaapi_cuda_get_proc_count();
    if (gpu_count != 2)
    {
      printf("invalid gpu settings\n");
      exit(-1);
    }

    double_type* arrays[2];
    const size_t total_size =
      CONFIG_ROW_COUNT * CONFIG_COL_COUNT * sizeof(double_type);

    double t0,t1, sum = 0.f;

    for (size_t iter = 0; iter < CONFIG_ITER_COUNT; ++iter)
    {
      t0 = kaapi_get_elapsedns();

      // prepare partitions
      ka::ThreadGroup threadgroup(3);

      threadgroup.begin_partition();

      // set kasid users to handle multi gpus
      const unsigned int first_kid = kaapi_cuda_get_first_kid();
      const unsigned int last_kid = first_kid + gpu_count;
      for (size_t cu_part = first_kid; cu_part < last_kid; ++cu_part)
      {
	const unsigned int kasid_user =
	  kaapi_cuda_get_kasid_user(cu_part - first_kid);
	threadgroup.force_kasid(cu_part, KAAPI_PROC_TYPE_CUDA, kasid_user);
      }

      // allocate the 2 arrays
      for (size_t count = 0; count < 2; ++count)
      {
	double_type* const array = (double_type*)kaapi_mem_alloc_host(total_size);
	printf("Spawn(%lx)\n", (uintptr_t)array);
	memset(array, 0, total_size);
	arrays[count] = array;
      }

      // spawn the dfg (r0, r1 the ranges):
      // cpu       gpu[0]             gpu[1]
      // ---------------------------------------
      // init(r0)
      // init(r1)
      //           addone(r0)         addone(r1)
      //           addranges(r0, r1)
      // fetch(r0)

      ka::range2d<double_type> r0
	(arrays[0], CONFIG_ROW_COUNT, CONFIG_COL_COUNT, CONFIG_COL_COUNT);
      ka::range2d<double_type> r1
	(arrays[1], CONFIG_ROW_COUNT, CONFIG_COL_COUNT, CONFIG_COL_COUNT);

      threadgroup.Spawn<TaskInit>(ka::SetPartition(0))(r0);
      threadgroup.Spawn<TaskInit>(ka::SetPartition(0))(r1);

      threadgroup.Spawn<TaskAddone>(ka::SetPartition(1))(r0);
      threadgroup.Spawn<TaskAddone>(ka::SetPartition(2))(r1);

      threadgroup.Spawn<TaskAddranges>(ka::SetPartition(2))(r0, r1);

      threadgroup.Spawn<TaskFetch>(ka::SetPartition(0))((uintptr_t)arrays[0], r0);

      threadgroup.end_partition();

      threadgroup.execute();

      t1 = kaapi_get_elapsedns();
      sum += (t1-t0)/1000; // us

      // check it
#if CONFIG_RANGE_CHECK
      printf(">>> checking range\n");
      double_type* const array = arrays[0];
      for (size_t row = 0; row < CONFIG_ROW_COUNT; ++row)
	for (size_t col = 0; col < CONFIG_COL_COUNT; ++col)
	{
	  const double_type value = array[row * CONFIG_COL_COUNT + col];
	  if (value != 2)
	  {
	    printf("invalid @(%u,%u) == %u\n", row, col, value);
	    row = CONFIG_ROW_COUNT - 1;
	    break;
	  }
	}
      printf("<<< checking range\n");
#endif

      for (size_t count = 0; count < 2; ++count)
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
