/*
** xkaapi
** 
** Created on Tue Mar 31 15:19:14 2009
** Copyright 2009 INRIA.
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

// this is a testing code for 2d ranges. it computes a scaled
// (averaged) output from a source image. it should not be
// used for benchmarking, since the kernel does not compute
// enough, and memory transfers are to small.


#include "kaapi++"
#include <algorithm>
#include <string.h>
#include <math.h>
#include <cuda.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/types.h>


// static configuration
// output must be square root of input dim
// otherwise the following code does not work 
#if 0
#define CONFIG_INPUT_DIM 1024
#define CONFIG_OUTPUT_DIM 32
#else
#define CONFIG_INPUT_DIM 4096
#define CONFIG_OUTPUT_DIM 64
#endif

// split is done on an output row basis
// otherwise, split done on a cell basis.
// non row mode should be used to tests 2d
// copy memory transfers since it bypass
// the contiguous optimization case.
#define CONFIG_USE_ROW 0

// compare with seq result
#define CONFIG_USE_CHECK 0

// time executions
#define CONFIG_USE_TIME 1


// image helpers

typedef uint64_t pixel_type;

static void create_images
(pixel_type*& in, pixel_type*& out_par, pixel_type*& out_seq)
{
  const size_t isize =
    CONFIG_INPUT_DIM * CONFIG_INPUT_DIM * sizeof(pixel_type);

  const size_t osize =
    CONFIG_OUTPUT_DIM * CONFIG_OUTPUT_DIM * sizeof(pixel_type);

  in = (pixel_type*)malloc(isize);

  for (size_t i = 0; i < (isize / sizeof(pixel_type)); ++i)
    in[i] = i & 0xff;

  out_par = (pixel_type*)malloc(osize);
  out_seq = (pixel_type*)malloc(osize);
}

static void destroy_images
(pixel_type* in, pixel_type* out_par, pixel_type* out_seq)
{
  free((void*)in);
  free((void*)out_par);
  free((void*)out_seq);
}

#if CONFIG_USE_CHECK
static int compare_images
(const pixel_type* a, const pixel_type* b)
{
  const pixel_type* const saved = a;

  size_t size = CONFIG_OUTPUT_DIM * CONFIG_OUTPUT_DIM;
  for (; size; --size, ++a, ++b)
  {
    if (!(*a == *b))
    {
      printf("invalid at %lx,%lu,%lu: %lu != %lu\n",
	     (uintptr_t)a,
	     (a - saved) / CONFIG_OUTPUT_DIM,
	     (a - saved) % CONFIG_OUTPUT_DIM,
	     *a, *b);
      return -1;
    }
  }
  return 0;
}
#endif // CONFIG_USE_CHECK

static pixel_type sum_block_pixels(const pixel_type* in)
{
  static const size_t w = CONFIG_OUTPUT_DIM;

  // substract w since adavanced in previous iteration
  static const size_t lda = CONFIG_INPUT_DIM - CONFIG_OUTPUT_DIM;

  pixel_type sum = 0;

  for (size_t i = 0; i < w; ++i, in += lda)
    for (size_t j = 0; j < w; ++j, ++in)
      sum += *in;

  return sum;
}

static void scale_image_seq(pixel_type* in, pixel_type* out)
{
  static const size_t stride = (CONFIG_OUTPUT_DIM - 1) * CONFIG_INPUT_DIM;

  static const size_t w = CONFIG_OUTPUT_DIM;
  static const pixel_type ww = CONFIG_OUTPUT_DIM * CONFIG_OUTPUT_DIM;

  // out[i] = sum(inblocks[i]) / ww;
  for (size_t i = 0; i < w; ++i, in += stride)
    for (size_t j = 0; j < w; ++j, in += w, ++out)
      *out = sum_block_pixels(in) / ww;
}

// scale parallel version

class ScaleWork
{
private:
  pixel_type* _in;
  pixel_type* _out;
  kaapi_workqueue_t _wq;

public:
  ScaleWork(pixel_type* in, pixel_type* out)
    : _in(in), _out(out)
  {
#if CONFIG_USE_ROW
    // output row count
    const size_t orows = CONFIG_OUTPUT_DIM;
    kaapi_workqueue_init(&_wq, 0, orows);
#else
    kaapi_workqueue_init
      (&_wq, 0, CONFIG_OUTPUT_DIM * CONFIG_OUTPUT_DIM);
#endif
  }

  bool extractPar(size_t& unit)
  {
    kaapi_workqueue_index_t i, j;
    if (kaapi_workqueue_steal(&_wq, &i, &j, 1)) return false;
    unit = i;
    return true;
  }

  bool extractSeq(size_t& unit)
  {
    kaapi_workqueue_index_t i, j;
    if (kaapi_workqueue_pop(&_wq, &i, &j, 1)) return false;
    unit = i;
    return true;
  }

  ka::array<2, pixel_type> unit_to_iarr(size_t unit)
  {
#if CONFIG_USE_ROW
    // unit an OUTPUT row
    pixel_type* const p = _in + unit * CONFIG_INPUT_DIM * CONFIG_OUTPUT_DIM;
    return ka::array<2, pixel_type>
      (p, CONFIG_OUTPUT_DIM, CONFIG_INPUT_DIM, CONFIG_INPUT_DIM);
#else
    // unit an OUTPUT cell

    const size_t row = unit / CONFIG_OUTPUT_DIM;
    const size_t col = unit % CONFIG_OUTPUT_DIM;

    pixel_type* const p =
      _in + (row * CONFIG_INPUT_DIM + col) * CONFIG_OUTPUT_DIM;

    return ka::array<2, pixel_type>
      (p, CONFIG_OUTPUT_DIM, CONFIG_OUTPUT_DIM, CONFIG_INPUT_DIM);
#endif
  }

  ka::array<1, pixel_type> unit_to_oarr(size_t unit)
  {
#if CONFIG_USE_ROW
    const size_t col_count = CONFIG_OUTPUT_DIM;
#else
    const size_t col_count = 1;
#endif
    pixel_type* const p = _out + unit * col_count;
    return ka::array<1, pixel_type>(p, col_count);
  }

  void split(ka::StealContext*, int, ka::Request*);
};

struct ScaleTask : public ka::Task<2>::Signature
<
  ka::R<ka::range2d<pixel_type> >,
  ka::W<ka::range1d<pixel_type> >
>{};

template<>
struct TaskBodyCPU<ScaleTask>
{
  void operator()
  (
   ka::range2d_r<pixel_type> in,
   ka::range1d_w<pixel_type> out
  )
  {
    // not implemented, assume gpu only
  }
};


#if CONFIG_USE_ROW

__global__ void ScaleKernelCuda
(const pixel_type* in, pixel_type* out)
{
  // out the output row to compute
  // inthe 2d matrix

  __shared__ pixel_type shared_sums[CONFIG_OUTPUT_DIM];

  // each cuda block works on it own input
  in = in + blockIdx.x * CONFIG_OUTPUT_DIM + threadIdx.x;

  // each thread sum its column in shared_sum[x]
  pixel_type local_sum = 0;
  for (size_t i = 0; i < CONFIG_OUTPUT_DIM; ++i, in += CONFIG_INPUT_DIM)
    local_sum += *in;
  shared_sums[threadIdx.x] = local_sum;

  syncthreads();

  if (threadIdx.x == 0)
  {
    const pixel_type ww = CONFIG_OUTPUT_DIM * CONFIG_OUTPUT_DIM;

    // reduce sums in out[blockDim.x]
    local_sum = 0;
    for (size_t i = 0; i < CONFIG_OUTPUT_DIM; ++i)
      local_sum += shared_sums[i];
    out[blockIdx.x] = local_sum / ww;
  }
}

static void ScaleKernelCpu
(const pixel_type* in, pixel_type* out)
{
  static const size_t w = CONFIG_OUTPUT_DIM;
  static const pixel_type ww = CONFIG_OUTPUT_DIM * CONFIG_OUTPUT_DIM;

  for (size_t i = 0; i < w; ++i, ++out, in += w)
    *out = sum_block_pixels(in) / ww;
}

#else // ! CONFIG_USE_ROW

__global__ void ScaleKernelCuda
(const pixel_type* in, pixel_type* out)
{
  // out the output row to compute
  // inthe 2d matrix

  __shared__ pixel_type shared_sums[CONFIG_OUTPUT_DIM];

  // each cuda block works on it own input
  in = in + threadIdx.x;

  // each thread sum its column in shared_sum[x]
  pixel_type local_sum = 0;
  for (size_t i = 0; i < CONFIG_OUTPUT_DIM; ++i, in += CONFIG_OUTPUT_DIM)
    local_sum += *in;
  shared_sums[threadIdx.x] = local_sum;

  syncthreads();

  if (threadIdx.x == 0)
  {
    const pixel_type ww = CONFIG_OUTPUT_DIM * CONFIG_OUTPUT_DIM;

    local_sum = 0;
    for (size_t i = 0; i < CONFIG_OUTPUT_DIM; ++i)
      local_sum += shared_sums[i];

    *out = local_sum / ww;
  }
}

static void ScaleKernelCpu
(const pixel_type* in, pixel_type* out)
{
  *out = sum_block_pixels(in) / (CONFIG_OUTPUT_DIM * CONFIG_OUTPUT_DIM);
}

#endif // CONFIG_USE_ROW

template<>
struct TaskBodyGPU<ScaleTask>
{
  void operator()
  (
   ka::gpuStream stream,
   ka::range2d_r<pixel_type> in,
   ka::range1d_w<pixel_type> out
  )
  {
    // 1 input block per SM
    // 1 thread per input block col

#if CONFIG_USE_ROW
    static const size_t block_count = CONFIG_OUTPUT_DIM;
#else
    static const size_t block_count = 1;
#endif

    const CUstream custream = (CUstream)stream.stream;
    ScaleKernelCuda<<<block_count, CONFIG_OUTPUT_DIM, 0, custream>>>
      (in.ptr(), out.ptr());
  }
};


void ScaleWork::split(ka::StealContext* sc, int nreq, ka::Request* req)
{
  size_t unit;

  for (; nreq; --nreq, ++req)
  {
    if (extractPar(unit) == false) return ;

    ka::array<2, pixel_type> iarr = unit_to_iarr(unit);
    ka::array<1, pixel_type> oarr = unit_to_oarr(unit);

    req->Spawn<ScaleTask>(sc)(iarr, oarr);
  }
}


static void scale_image_par(pixel_type* in, pixel_type* out)
{
  ka::StealContext* sc;

  ScaleWork work(in, out);

  sc = ka::TaskBeginAdaptive
  (
   KAAPI_SC_CONCURRENT | KAAPI_SC_NOPREEMPTION,
   &ka::WrapperSplitter<ScaleWork, &ScaleWork::split>,
   &work
  );

  size_t unit;

  while (work.extractSeq(unit))
  {
    ka::array<2, pixel_type> iarr = work.unit_to_iarr(unit);
    ka::array<1, pixel_type> oarr = work.unit_to_oarr(unit);
    ScaleKernelCpu(iarr.ptr(), oarr.ptr());
  }

  ka::TaskEndAdaptive(sc);
}


// main task
struct doit
{
  void operator()(int argc, char** argv )
  {
    pixel_type* in;
    pixel_type* out_seq;
    pixel_type* out_par;

#if CONFIG_USE_TIME
    double t0, t1;
#endif

    create_images(in, out_par, out_seq);

#if CONFIG_USE_TIME
    t0 = kaapi_get_elapsedtime();
#endif

    scale_image_seq(in, out_seq);

#if CONFIG_USE_TIME
    t1 = kaapi_get_elapsedtime();
    std::cout << t1 - t0 << std::endl; // seconds
#endif

    for (size_t iter = 0; iter < 10; ++iter)
    {
#if CONFIG_USE_TIME
      t0 = kaapi_get_elapsedtime();
#endif

      scale_image_par(in, out_par);

#if CONFIG_USE_TIME
      t1 = kaapi_get_elapsedtime();
      std::cout << t1 - t0 << std::endl; // seconds
#endif

#if CONFIG_USE_CHECK
      if (compare_images(out_par, out_seq))
	printf("invalid\n");
#endif
    }

    destroy_images(in, out_par, out_seq);
  }
};


int main(int argc, char** argv)
{
  try
  {
    ka::Community com = ka::System::join_community(argc, argv);
    ka::SpawnMain<doit>()(argc, argv); 
    com.leave();
    ka::System::terminate();
  }
  catch (const std::exception& E)
  {
    ka::logfile() << "Catch : " << E.what() << std::endl;
  }
  catch (...)
  {
    ka::logfile() << "Catch unknown exception: " << std::endl;
  }

  return 0;
}
