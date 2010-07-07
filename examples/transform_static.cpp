#include "kaapi++"
#include <algorithm>
#include "random.h"

struct Op2 {
    Op2(){ };
    /* __device__ __host__ */ double operator()(double a){
      return  2*a;
    }
};

extern /* __global__ */ void transform_kernel( const double* beg, const double* end, double* out );

// --------------------------------------------------------------------
struct TransformTask: public ka::Task<3>::Signature< 
        ka::R<double>,
        ka::R<double>,
        ka::W<double>
> {};
template<> struct TaskFormat<TransformTask> {
  typedef ka::Tuple<
    ka::R<>, ka::ArrayType<1, double>, 
    ka::W<>, ka::ArrayType<1, double>
  > View;
  static void map( View& view,
    ka::pointer_r<double> beg, ka::pointer_r<double> end, ka::pointer_w<double> out
  )
  {
    view.r1.bind( beg, end-beg );
    view.r2.bind( out, end-beg );
  }
  static void umap( const View& view,
    ka::pointer_r<double>& beg, ka::pointer_r<double>& end, ka::pointer_w<double>& out
  )
  {
    beg = view.r1.addr();
    end = beg + view.r1.count();
    out = view.r2.addr();
    kaapi_assert( view.r2.count() == view.r1.count() );
  }
};


template<> struct TaskBodyCPU<TransformTask> {
  void operator() ( ka::pointer_r<double> beg, ka::pointer_r<double> end, ka::pointer_w<double> out )
  {
    std::transform( beg, end, (double*)out, Op2() );
  }
};
static ka::RegisterBodyCPU<TransformTask> dummy_object_TransformTask;


template<> struct TaskBodyGPU<TransformTask> {
  void operator() ( ka::gpuStream stream, ka::pointer_r<double> beg, ka::pointer_r<double> end, ka::pointer_w<double> out )
  {
#if 0
    blocks = ...
    threads = ...
    transform_kernel<<<blocks, threads, 0, stream>>>( beg, end, out, Op2() );
#endif
  }
};


// --------------------------------------------------------------------
struct KernelTransform {
  void operator() ( 
                    int p,
                    int n,
                    double* input,
                    double* output
  )
  {
    int bloc_size = n/p;
    double* beg = input;
    double* end = beg+bloc_size;
    for (int i=0; i<p; ++i)
    {
      ka::Spawn<TransformTask>( ka::SetPartition(i) )( beg, end, output );
      beg = end;
      output += bloc_size;
      end = beg+bloc_size;
      if (i == p-1) end = input+n; 
    }
  }
};


// --------------------------------------------------------------------
/** Main Task, only executed on one process.
*/
struct doit {
  void operator()(int argc, char** argv )
  {   
    int p, n, iter;
    double t0, t1;

    if (argc >1) n = atoi(argv[1]);
    else n = 10000;
    if (argc >2) iter = atoi(argv[2]);
    else iter = 1;

    t0 = kaapi_get_elapsedtime();
    double* input  = new double[n];
    double* output = new double[n];
    t1 = kaapi_get_elapsedtime();

    std::cout << "Time allocate:" << t1-t0 << std::endl;
    Random random(42 + iter);

    // number of partitions = X*Y*Z
    ka::ThreadGroup threadgroup( p = kaapi_getconcurrency() );

    t0 = kaapi_get_elapsedtime();
    for(int i = 0; i < n; ++i) 
    {
      input[i] = i; //random.genrand_real3();
      output[i] = 0;
    }
    t1 = kaapi_get_elapsedtime();
    std::cout << "Time init:" << t1-t0 << std::endl;

    threadgroup.ForEach<KernelTransform>()
      ( ka::counting_iterator<int>(0), ka::counting_iterator<int>(iter) ) /* iteration space */
      ( p, n, input, output );                      /* args for the kernel */

  }
};


// --------------------------------------------------------------------
/** Main of the program
*/
int main( int argc, char** argv ) 
{
  try {
    ka::Community com = ka::System::join_community( argc, argv );
    ka::SpawnMain<doit>()(argc, argv); 
    com.leave();
    ka::System::terminate();
  }
  catch (ka::Exception& ex) {
    std::cerr << "[main] catch exception: " << ex.what() << std::endl;
  }
  catch (...) {
    std::cerr << "[main] catch unknown exception" << std::endl;
  }
  return 0;
}

