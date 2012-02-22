
#include <iostream>
#include <cstdio>
#include <algorithm>
#include <cuda.h>

#include "kaapi++"

#define	    BLOCK_SIZE		    256

typedef float double_type;

static int global_block_size = 0;

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

    dim3 threads( BLOCK_SIZE, 1 );
    unsigned int grid_size= (range_size+BLOCK_SIZE-1)/BLOCK_SIZE;
    dim3 grid( (grid_size < 65536) ? grid_size : 32768, 1 );
    unsigned int n_block= range_size/(BLOCK_SIZE*grid.x);
    transform_kernel<<< grid, threads, 0, custream >>>( range.begin(),
	    range_size, n_block, transform_gpu(1.f) );
  }
};

// CPU implementation
template<> struct TaskBodyCPU<TaskAddone>
{
  void operator() (ka::range1d_rw<double_type> range)
  {
    const size_t range_size = range.size();
    fprintf(stdout,"cpuTask (%p, %lu)\n",
	   range.begin(), range.size() ); 
    fflush(stdout);

    std::transform( range.begin(), range.begin()+range_size,
	    range.begin(),
	    transform_gpu(1.f) );
  }
};

struct TaskAddoneMain : public ka::Task<1>::Signature<ka::RW<ka::range1d<double_type> > > {};

template<> struct TaskBodyCPU<TaskAddoneMain>
{
  void operator() (ka::range1d_rw<double_type> range) {
    const size_t range_size = range.size();

    fprintf(stdout,"cpuTaskMAIN (%lx, %lu)\n",
	   (uintptr_t)range.begin(), range.size());
    fflush(stdout);

    for( size_t range_pos= 0; range_pos < range_size; range_pos+=
	    global_block_size ) {
	ka::range1d<double_type> l( range.begin()+range_pos, global_block_size );
	ka::Spawn<TaskAddone>()( l );
    }
  }
};

// main task
struct doit {
  void operator()(int argc, char** argv)
  {
    size_t size = 100000;
    int block_size = 64;
    int verif = 0;

    if (argc >1)
	size = atoi( argv[1] );
    if( argc > 2 )
	block_size = atoi( argv[2] );
    if( argc > 3 )
	verif = atoi( argv[3] );
    
    double_type* array = (double_type*) calloc(size, sizeof(double_type));
    kaapi_memory_register( array, size*sizeof(double_type) );
    fprintf(stdout,"MAIN array=%p\n", array);fflush(stdout);
    global_block_size = block_size;

      // initialize, apply, check
      for (size_t i = 0; i < size; ++i)
        array[i] = 0.f;
        
    double t0 = kaapi_get_elapsedtime();

	ka::range1d<double_type> range(array, size);
	ka::Spawn<TaskAddoneMain>( ka::SetStaticSched() )( range );
	ka::Sync();
	ka::MemorySync();

    double t1 = kaapi_get_elapsedtime();
    double tdelta = t1 - t0;

      if( verif ) {
	  for (size_t i = 0; i < size; ++i) {
	    if( array[i] != 1.f ) {
	      std::cout << "ERROR invalid @" << i << " == " << array[i] << std::endl;
	      break ;
	    }
	  }
      }

    fprintf( stdout, "transform_static %d %d %d %.10f\n", size, block_size,
		    kaapi_getconcurrency(), tdelta );
    fflush(stdout);
    //std::cout << "Done " << sum/100 << " (ms)" << std::endl;
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
