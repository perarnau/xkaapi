
#include <stdint.h>
#include <sys/types.h>
#include <cuda.h>
#include "kaapi++"

#include <unistd.h>

typedef float double_type;

struct TaskHello: public ka::Task<1>::Signature< int >{};

template<> struct TaskBodyCPU<TaskHello>
{
    void operator()( int i )
    {
	fprintf( stdout,"HELLO CPU\n" );
	fflush(stdout);
    }
};

template<> struct TaskBodyGPU<TaskHello>
{
    void operator()( ka::gpuStream stream, int i )
    {
	fprintf( stdout,"HELLO CUDA\n" );
	fflush(stdout);
    }
};

///////////////////////////////////////////////////////////////////////////////

// task signature
struct TaskAddone : public ka::Task<1>::Signature<ka::RW<ka::range1d<double_type> > > {};

// cuda kernel
__global__ void addone(double_type* array, unsigned int size)
{
  const unsigned int per_thread = size / blockDim.x;
  unsigned int i = threadIdx.x * per_thread;

  unsigned int j = size;
  if (threadIdx.x != (blockDim.x - 1)) j = i + per_thread;

  for (; i < j; ++i) ++array[i];
}

template<> struct TaskBodyGPU<TaskAddone>
{
  void operator()(ka::gpuStream stream, ka::range1d_rw<double_type> range)
  {
    fprintf(stdout,"gpuTask (%lx, %lu)\n",
	   (uintptr_t)range.begin(), range.size() );
    fflush(stdout);

    const CUstream custream = (CUstream)stream.stream;
    addone<<<1, 256, 0, custream>>>(range.begin(), range.size());
  }
};

// cpu implementation
template<> struct TaskBodyCPU<TaskAddone>
{
  void operator() (ka::range1d_rw<double_type> range)
  {
    const size_t range_size = range.size();
    fprintf(stdout,"cpuTask (%lx, %lu)\n",
	   (uintptr_t)range.begin(), range.size() ); 
    fflush(stdout);
    for (size_t i = 0; i < range_size; ++i)
      range[i] += 1;
  }
};

struct TaskAddoneMain : public ka::Task<1>::Signature<ka::RW<ka::range1d<double_type> > > {};

// cpu implementation
template<> struct TaskBodyCPU<TaskAddoneMain>
{
  void operator() (ka::range1d_rw<double_type> range)
  {
    const size_t range_size = range.size();
    const size_t nb_size = range_size / 8;

    fprintf(stdout,"cpuTaskMAIN (%lx, %lu)\n",
	   (uintptr_t)range.begin(), range.size());
    fflush(stdout);

    for( size_t pos= 0; pos < range_size; pos+= nb_size ){
    	ka::range1d<double_type> l(range.begin()+pos, nb_size);
	ka::Spawn<TaskAddone>()( l );
    }
  }
};

// main task
struct doit {
  void operator()(int argc, char** argv)
  {
    double t0,t1;
    double sum = 0.f;
    size_t size = 100000;
    if (argc >1) size = atoi(argv[1]);
    
    double_type* array = (double_type*) calloc(size, sizeof(double_type));
    fprintf(stdout,"MAIN array=%p\n", array);fflush(stdout);

    for (int iter = 0; iter < 1; ++iter)
    {
      // initialize, apply, check
      for (size_t i = 0; i < size; ++i)
        array[i] = 0.f;
        
      t0 = kaapi_get_elapsedns();

      // fork the root task
      ka::range1d<double_type> range(array, size);
       ka::Spawn<TaskAddoneMain>()(range);
      ka::Sync();

      t1 = kaapi_get_elapsedns();

      sum += (t1-t0)/1000; // ms

      for (size_t i = 0; i < size; ++i)
      {
        if (array[i] != 1.f)
        {
          std::cout << "ERROR invalid @" << i << " == " << array[i] << std::endl;
          break ;
        }
      }
    }

    std::cout << "Done " << sum/100 << " (ms)" << std::endl;
  }
};


/* main entry point : Kaapi initialization
*/
int main(int argc, char** argv)
{
  try {
    /* Join the initial group of computation : it is defining
       when launching the program by a1run.
    */
    ka::Community com = ka::System::join_community( argc, argv );
    
    /* Start computation by forking the main task */
    ka::SpawnMain<doit>()(argc, argv); 
    
    /* Leave the community: at return to this call no more athapascan
       tasks or shared could be created.
    */
    com.leave();

    /* */
    ka::System::terminate();
  }
  catch (const std::exception& E) {
    ka::logfile() << "Catch : " << E.what() << std::endl;
  }
  catch (...) {
    ka::logfile() << "Catch unknown exception: " << std::endl;
  }
  
  return 0;
}
