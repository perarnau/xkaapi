
#include <iostream>
#include <algorithm>
#include <cuda.h>

#include "kaapi++"

typedef float double_type;
#define	    TRANSFORM_GRAIN_SIZE    256
#define	    BLOCK_SIZE		    256

template<class Op>
__global__ void transform_kernel( double_type *y, const unsigned int N,
		const unsigned int n_block,
		Op op )
{
	int index= (blockDim.x * blockIdx.x + threadIdx.x)*n_block;
	int i;
	for( i= 0; i < n_block; i++ )
		if( (index+i) < N )
			y[index+i]= op( y[index+i] );
}

struct transform_gpu
{
    const double_type a;

    transform_gpu (double_type _a) : a(_a) {}

    __host__ __device__
        double_type operator()( const double_type& y ) const { 
            return y+a;
        }
};

// task signature
struct TaskAddone : public ka::Task<1>::Signature<ka::RW<ka::range1d<double_type> > > {};

template<> struct TaskBodyGPU<TaskAddone>
{
  void operator()( ka::gpuStream stream, ka::range1d_rw<double_type> range )
  {
    const CUstream custream = (CUstream)stream.stream;
    const size_t range_size = range.size();

    fprintf(stdout,"gpuTask (%p, %lu)\n",
	   range.begin(), range.size() );
    fflush(stdout);

    if( TRANSFORM_GRAIN_SIZE >= range.size() ) {
	dim3 threads( BLOCK_SIZE, 1 );
	unsigned int grid_size= (range_size+BLOCK_SIZE-1)/BLOCK_SIZE;
	dim3 grid( (grid_size < 65536) ? grid_size : 32768, 1 );
	unsigned int n_block= range_size/(BLOCK_SIZE*grid.x);
	transform_kernel<<< grid, threads, 0, custream >>>( range.begin(),
		range_size, n_block, transform_gpu(1.f) );
    } else {
	const size_t new_size= range_size/2;
	ka::range1d<double_type> range1( range.begin(), new_size );
	ka::range1d<double_type> range2(
		range.begin()+new_size*sizeof(double_type),
		range_size-new_size );
	ka::Spawn<TaskAddone>() ( range1 );
	ka::Spawn<TaskAddone>() ( range2 );
	ka::Sync();
    }
  }
};

// cpu implementation
template<> struct TaskBodyCPU<TaskAddone>
{
  void operator() (ka::range1d_rw<double_type> range)
  {
    const size_t range_size = range.size();
    fprintf(stdout,"cpuTask (%p, %lu)\n",
	   range.begin(), range.size() ); 
    fflush(stdout);

    if( TRANSFORM_GRAIN_SIZE >= range_size ) {
	std::transform( range.begin(), range.begin()+range_size,
		range.begin(),
		transform_gpu(1.f) );
    } else {
	const size_t new_size= range_size/2;
	ka::range1d<double_type> range1( range.begin(), new_size );
	ka::range1d<double_type> range2( range.begin()+new_size, range_size-new_size );
	ka::Spawn<TaskAddone>() ( range1 );
	ka::Spawn<TaskAddone>() ( range2 );
	ka::Sync();
    }
  }
};

struct TaskAddoneMain : public ka::Task<1>::Signature<ka::RW<ka::range1d<double_type> > > {};

// cpu implementation
template<> struct TaskBodyCPU<TaskAddoneMain>
{
  void operator() (ka::range1d_rw<double_type> range)
  {
    const size_t range_size = range.size();
    fprintf(stdout,"cpuTaskMAIN (%lx, %lu)\n",
	   (uintptr_t)range.begin(), range.size());
    fflush(stdout);

    ka::range1d<double_type> l(range.begin(), range_size);
    ka::Spawn<TaskAddone>()( l );
	ka::Sync();
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
    kaapi_memory_synchronize();

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
