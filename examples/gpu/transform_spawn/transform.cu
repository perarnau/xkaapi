/*
** xkaapi
** 
** Created on Tue Mar 31 15:19:14 2009
** Copyright 2009,2010,2011,2012 INRIA.
**
** Contributors :
** thierry.gautier@inrialpes.fr
** fabien.lementec@imag.fr
** 
** This software is a computer program whose purpose is to execute
** multithreaded computation with data flow synchronization between
** threads.
** 
** This software is governed by the CeCILL-C license under French law
** and abiding by the rules of distribution of free software.  You can
** use, modify and/ or redistribute the software under the terms of
** the CeCILL-C license as circulated by CEA, CNRS and INRIA at the
** following URL "http://www.cecill.info".
** 
** As a counterpart to the access to the source code and rights to
** copy, modify and redistribute granted by the license, users are
** provided only with a limited warranty and the software's author,
** the holder of the economic rights, and the successive licensors
** have only limited liability.
** 
** In this respect, the user's attention is drawn to the risks
** associated with loading, using, modifying and/or developing or
** reproducing the software by the user in light of its specific
** status of free software, that may mean that it is complicated to
** manipulate, and that also therefore means that it is reserved for
** developers and experienced professionals having in-depth computer
** knowledge. Users are therefore encouraged to load and test the
** software's suitability as regards their requirements in conditions
** enabling the security of their systems and/or data to be ensured
** and, more generally, to use and operate it in the same conditions
** as regards security.
** 
** The fact that you are presently reading this means that you have
** had knowledge of the CeCILL-C license and that you accept its
** terms.
** 
*/
#include <stdint.h>
#include <sys/types.h>
#include <cuda.h>
#include "kaapi++"

#include <unistd.h>
static void waitabit(void)
{ usleep(10000); }

// missing decls
typedef uintptr_t kaapi_mem_addr_t;
extern "C" void kaapi_mem_delete_host_mappings(kaapi_mem_addr_t, size_t);
typedef float double_type;

// task signature
struct TaskAddone : public ka::Task<1>::Signature
<ka::RW<ka::range1d<double_type> > > {};

// cuda kernel
__global__ void addone(double_type* array, unsigned int size)
{
  const unsigned int per_thread = size / blockDim.x;
  unsigned int i = threadIdx.x * per_thread;

  unsigned int j = size;
  if (threadIdx.x != (blockDim.x - 1)) j = i + per_thread;

  for (; i < j; ++i) ++array[i];
}

// cpu implementation
template<> struct TaskBodyCPU<TaskAddone>
{
#if 0 // todo_recursive
  void operator() (ka::range1d_rw<double_type> range)
  {
    printf("cpuTask %u\n", kaapi_get_self_kid());

    const size_t range_size = range.size();

    // reached the leaf size
    if (range_size < 512)
    {
      for (size_t i = 0; i < range_size; ++i)
	range[i] += 1;
      return ;
    }

    // recurse by spawning 2 tasks with 1/2 range each

    double_type* const lend = range.begin() + range_size / 2;

    ka::range1d<double_type> l(range.begin(), lend);
    ka::range1d<double_type> r(lend, range_size - l.size());

    ka::Spawn<TaskAddone>()(l);
    ka::Spawn<TaskAddone>()(r);
  }
#else // todo_recursive
  void operator() (ka::range1d_rw<double_type> range)
  {
    printf("cpuTask %u\n", kaapi_get_self_kid());

    const size_t range_size = range.size();
    for (size_t i = 0; i < range_size; ++i)
      range[i] += 1;
  }
#endif // todo_recursive

};

// gpu implementation

extern "C" CUstream kaapi_cuda_kernel_stream(void);

template<> struct TaskBodyGPU<TaskAddone>
{
  void operator()(ka::gpuStream stream, ka::range1d_rw<double_type> range)
  {
    printf("[%u] gpuTask (%lx, %lu)\n",
	   kaapi_get_self_kid(),
	   (uintptr_t)range.begin(), range.size());

    const CUstream custream = (CUstream)stream.stream;
    addone<<<1, 256, 0, custream>>>(range.begin(), range.size());
  }

  void operator()(ka::range1d_rw<double_type> range)
  {
    // helper to bypass a bug in code generation
    ka::gpuStream gpustream
      ((kaapi_gpustream_t)kaapi_cuda_kernel_stream());
    (*this)(gpustream, range);
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
    
    double_type* array = new double_type[size];

    for (int iter = 0; iter < 1; ++iter)
    {
      // initialize, apply, check
      for (size_t i = 0; i < size; ++i)
        array[i] = 0.f;
        
      t0 = kaapi_get_elapsedns();

      // fork the root task
      ka::range1d<double_type> range(array, size);
      // ka::Spawn<TaskAddone>(ka::SetStaticSched())(range);
      ka::Spawn<TaskAddone>()(range);
      waitabit(); // gpu task scheduled
      ka::Sync();

      kaapi_mem_delete_host_mappings
	((kaapi_mem_addr_t)array, sizeof(double_type) * size);

      t1 = kaapi_get_elapsedns();

      sum += (t1-t0)/1000; // ms

      for (size_t i = 0; i < size; ++i)
      {
        if (array[i] != 1.f)
        {
          std::cout << "invalid @" << i << " == " << array[i] << std::endl;
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
