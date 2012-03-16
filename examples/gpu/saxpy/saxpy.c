
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stddef.h>
#include <unistd.h>
#include <sys/time.h>

#include "kaapi.h"
#include "kaapi_cuda_func.h"

#define CONFIG_KERNEL_DIM 256

int compareL2fe( const float* reference, const float* data,
                const unsigned int len, const float epsilon ) 
{
    float error = 0;
    float ref = 0;

    for( unsigned int i = 0; i < len; ++i) {

        float diff = reference[i] - data[i];
        error += diff * diff;
        ref += reference[i] * reference[i];
    }

    float normRef = sqrtf(ref);
    if (fabs(ref) < 1e-7) {
#ifdef _DEBUG
        fprintf( stdout, "ERROR, reference l2-norm is 0\n");
#endif
        return 0;
    }
    float normError = sqrtf(error);
    error = normError / normRef;
#ifdef _DEBUG
    if( ! (error < epsilon)) 
        fprintf( stdout, "ERROR, l2-norm error %f is greater than epsilon %f\n",
		 error, epsilon );
#endif

    return ( error < epsilon );
}

void check( float A, float *x, float *y, float *ref_y, unsigned int N )
{
	int result;
	int i;

	for( i = 0; i < N; i++ ) {
		ref_y[i] += A * x[i];
	}
	result= compareL2fe( ref_y, y, N, 1e-6f );
	if( result == 0 ) {
		fprintf( stdout, "ERROR\n" );
		//fprintf( stdout, "%f %f\n", y[N-1], ref_y[N-1] );
	}
}

void randomInit( float* data, int size )
{
    for (int i = 0; i < size; ++i)
        data[i] = rand() / (float)RAND_MAX;
}

typedef struct range {
	kaapi_access_t x;
	kaapi_access_t y;
	float A;
	unsigned int i;
	unsigned int j;
} range_t;

typedef struct task_work {
	volatile long lock;
	range_t range;
} task_work_t;

static inline unsigned int get_range_size( const range_t* range )
{
	return (range->j - range->i);
}

/* task formatting */

#define PARAM_COUNT 5

static size_t get_count_params(const struct kaapi_format_t* f, const void* p)
{ return PARAM_COUNT; }

static void* get_off_param (const struct kaapi_format_t* f,
		unsigned int i, const void* p)
{
  static const kaapi_offset_t param_offsets[PARAM_COUNT] =
  {
    offsetof(task_work_t, range.x),
    offsetof(task_work_t, range.y),
    offsetof(task_work_t, range.A),
    offsetof(task_work_t, range.i),
    offsetof(task_work_t, range.j)
  };
  return (void*)((uintptr_t)p + param_offsets[i]);
}

static kaapi_access_mode_t get_readwrite_mode_param
(const struct kaapi_format_t* f, unsigned int i, const void* p)
{
  static const kaapi_access_mode_t modes[PARAM_COUNT] =
    { KAAPI_ACCESS_MODE_R, KAAPI_ACCESS_MODE_RW,
	   KAAPI_ACCESS_MODE_V, KAAPI_ACCESS_MODE_V, KAAPI_ACCESS_MODE_V };
  return modes[i];
}

static kaapi_access_t get_access_param
(const struct kaapi_format_t* f, unsigned int i, const void* p)
{
  task_work_t* const w = (task_work_t*)p;

  kaapi_access_t a;
  switch( i ) {
	  case 0:
	  a.data = (float*)w->range.x.data + w->range.i;
	  a.version = w->range.x.version;
	  break;
	  case 1:
	  a.data = (float*)w->range.y.data + w->range.i;
	  a.version = w->range.y.version;
	  break;
  }
  return a;
}

static void set_access_param
(const struct kaapi_format_t* f, unsigned int i, void* p, const kaapi_access_t* a)
{
  task_work_t* const w = (task_work_t*)p;

  switch( i ) {
	  case 0:
	  {
	  w->range.x.data = a->data;
	  w->range.x.version = a->version;
	  break;
	  }
	  case 1:
	  {
	  w->range.y.data = a->data;
	  w->range.y.version = a->version;
	  break;
	  }
  }
}

static const struct kaapi_format_t* get_fmt_param
(const struct kaapi_format_t* f, unsigned int i, const void* p)
{
#define kaapi_mem_format kaapi_ulong_format
  const struct kaapi_format_t* formats[PARAM_COUNT] =
    { kaapi_mem_format, kaapi_mem_format,
	    kaapi_float_format, kaapi_uint_format, kaapi_uint_format };
  return formats[i];
}

static size_t get_size_param
(const struct kaapi_format_t* f, unsigned int i, const void* p)
{
  switch (i)
  {
	  case 0: /* range */
	  case 1:
	  {
	    const task_work_t* const work = (task_work_t*)p;
	    return (size_t)get_range_size(&work->range) * sizeof(float);
	    break;
	  }
	  case 2: return sizeof(float);

	default: return sizeof(unsigned int);
  }
}

static task_work_t* alloc_work( kaapi_thread_t* thread )
{
	task_work_t* const work = kaapi_thread_pushdata_align(
			thread, sizeof(task_work_t), 8 );

	work->range.x.data= NULL;
	work->range.y.data= NULL;
	work->range.A= 0;
	work->range.i= 0;
	work->range.j= 0;

	return work;
}

static void lock_work( task_work_t* work )
{
	while (1) {
		if (__sync_bool_compare_and_swap(&work->lock, 0, 1))
			break;
	}
}

static void unlock_work( task_work_t* work )
{
	work->lock = 0;
	__sync_synchronize();
}

static void create_range( range_t* range, 
	float *x, float *y, const float A, unsigned int i, unsigned int j )
{
	kaapi_access_init( &range->x, (void*)x );
	kaapi_access_init( &range->y, (void*)y );
	range->A= A;
	range->i= i;
	range->j= j;
}

static void saxpy_cuda_entry ( CUstream stream, void* arg,
		kaapi_thread_t* thread	)
{
	task_work_t* const work = (task_work_t*)arg;
	range_t* const range = &work->range;
	CUdeviceptr d_x = (CUdeviceptr)(uintptr_t)range->x.data;
	CUdeviceptr d_y = (CUdeviceptr)(uintptr_t)range->y.data;

	fprintf( stdout, "saxpy_cuda [ %d %d [\n", range->i, range->j );
	fflush(stdout);

	{
#define THREAD_DIM_X CONFIG_KERNEL_DIM
	static const kaapi_cuda_dim3_t tdim = {THREAD_DIM_X, 1, 1};
	static const kaapi_cuda_dim2_t bdim = {1, 1};

	kaapi_cuda_func_t fn;

	kaapi_cuda_func_init(&fn);

#define STERN "saxpy_kernel"
	kaapi_cuda_func_load_ptx(&fn, STERN ".ptx", "saxpy_kernel");

	/* since base points to the start of the range */
	kaapi_cuda_func_push_ptr(&fn, d_x);
	kaapi_cuda_func_push_ptr(&fn, d_y);
	kaapi_cuda_func_push_float(&fn, range->A);
	kaapi_cuda_func_push_uint(&fn, 0);
	kaapi_cuda_func_push_uint(&fn, range->j - range->i);

	kaapi_cuda_func_call_async(&fn, stream, &bdim, &tdim);

	kaapi_cuda_func_unload_ptx(&fn);
	}
}

static void saxpy_cpu_entry( void *arg, kaapi_thread_t* thread )
{
	range_t* const range = &((task_work_t*)arg)->range;
	float* const x = range->x.data;
	float* const y = range->y.data;
	unsigned int i;

	fprintf( stdout, "saxpy_cpu [ %d %d [\n", range->i, range->j );
	fflush(stdout);
	/* working here */
	for( i= range->i; i < range->j; i++ )
		y[i] += x[i] * range->A;

}

static void register_saxpy_task_format( void )
{
	struct kaapi_format_t* const format= kaapi_format_allocate();

	kaapi_format_taskregister_func (
		format, NULL, "saxpy_task", sizeof(task_work_t),
		get_count_params,
		get_readwrite_mode_param,
		get_off_param,
		get_access_param,
		set_access_param,
		get_fmt_param,
		get_size_param
	);

	kaapi_format_taskregister_body
		( format, saxpy_cpu_entry, KAAPI_PROC_TYPE_CPU );
	kaapi_format_taskregister_body
		( format, (kaapi_task_body_t)saxpy_cuda_entry,
		  KAAPI_PROC_TYPE_CUDA );
}

static int split_range( range_t* sub, range_t* range, unsigned int size )
{
	const unsigned int range_size = get_range_size(range);
	if (range_size == 0) return -1;

	/* succeed even if size too large */
	if (range_size < size)
		size = range_size;

	sub->i = range->i;
	sub->j = range->i + size;
	range->i += size;

	return 0;
}

static int next_seq( task_work_t* work )
{
	float* const x= (float*)work->range.x.data;
	float* const y= (float*)work->range.y.data;
	float const A= work->range.A;

	int res;
	range_t subrange;
	unsigned int i;

	lock_work( work );
	res= split_range( &subrange, &work->range, 512 );
	unlock_work( work );

	if( res == -1) return -1;

#if 0
	fprintf( stdout, "saxpy_seq [ %d %d [\n", subrange.i, subrange.j );
	fflush(stdout);
#endif

	/* working here */
	for( i= subrange.i; i < subrange.j; i++ )
		y[i] += x[i] * A;

	return 0;
}

static int steal_range(range_t* sub, range_t* range, unsigned int size)
{
	if (get_range_size(range) < size)
		return -1;

	sub->j = range->j;
	sub->i = range->j - size;
	range->j = sub->i;

	return 0;
}

static int splitter( kaapi_stealcontext_t* sc, int reqcount,
		kaapi_request_t* reqs, void* arg)
{
	task_work_t* const vwork = (task_work_t*)arg;
	unsigned int rangesize;
	unsigned int unitsize;
	range_t subrange;
	int stealres = -1;
	int repcount = 0;

	lock_work(vwork);

	rangesize = get_range_size(&vwork->range);
	if (rangesize > 512) {
		unitsize = rangesize / (reqcount + 1);
		if (unitsize == 0) {
			unitsize = 512;
			reqcount = rangesize / 512;
		}

		stealres = steal_range(&subrange, &vwork->range,
				unitsize * reqcount);
	}

	unlock_work(vwork);

	if (stealres == -1)
		return 0;

	for (; reqcount > 0; ++reqs)
	{
		task_work_t* const twork = kaapi_reply_init_adaptive_task
			(sc, reqs, (kaapi_task_body_t)saxpy_cpu_entry,
			 sizeof(task_work_t), NULL);

		twork->lock = 0;
		kaapi_access_init(&twork->range.x, vwork->range.x.data);
		kaapi_access_init(&twork->range.y, vwork->range.y.data);
		twork->range.A= vwork->range.A;
		split_range(&twork->range, &subrange, unitsize);

		kaapi_reply_pushhead_adaptive_task(sc, reqs);

		--reqcount;
		++repcount;
	}

	return repcount;
}

void saxpy( float A, float *x, float *y, unsigned int N )
{
	kaapi_thread_t* const thread= kaapi_self_thread();
	task_work_t *work;
	kaapi_stealcontext_t* ksc;

	register_saxpy_task_format();
	work= alloc_work( thread );
	create_range( &work->range, x, y, A, 0, N );

	ksc= kaapi_task_begin_adaptive
		(thread, KAAPI_SC_CONCURRENT, splitter, work);

	while( next_seq(work) != -1 ) ;

	kaapi_task_end_adaptive( thread, ksc );

  /* wait for thieves */
  kaapi_sched_sync();
}

int main( int argc, char *argv[] )
{
	float *x, *y, *ref_y;
	unsigned int nelem= CONFIG_KERNEL_DIM * 10000;
	unsigned long mem_size;

	if( argc > 1 )
		nelem=  atoi(argv[1]);
	mem_size= nelem * sizeof(float);
	x= (float*) malloc( mem_size );
	y= (float*) malloc( mem_size );
	ref_y= (float*) malloc( mem_size );
	randomInit( x, nelem );
	randomInit( y, nelem );
	memcpy( ref_y, y, mem_size );

	kaapi_init();
	saxpy( 2.0, x, y, nelem );
	check( 2.0, x, y, ref_y, nelem );
	kaapi_finalize();
	free( x );
	free( y );
	free( ref_y );
}

