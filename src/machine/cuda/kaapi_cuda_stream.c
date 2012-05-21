/*
** xkaapi
** 
**
** Copyright 2009 INRIA.
**
** Contributors :
**
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
#include <stdlib.h>
#include <stddef.h>
#include <cuda.h>
#include <sys/types.h>
#include <errno.h>
#include <stdarg.h>

#include <cuda_runtime_api.h>

#include "kaapi_impl.h"
#include "kaapi_cuda_stream.h"

/*
*/
static int kaapi_cuda_request_destroy( kaapi_cuda_request_t* req)
{
  if ( (req->status.state == KAAPI_CUDA_REQUEST_PUSH) 
    || (req->status.state == KAAPI_CUDA_REQUEST_TERM) )
    return EINVAL;

#if CONFIG_USE_EVENT
  cudaEventDestroy( req->event );
#endif
  req->status.state = KAAPI_CUDA_REQUEST_TERM;
  return 0;
}


/* Create a CUDA fifo stream.
   \retval 0 in case of success
   \retval else an error code
*/
static inline int kaapi_cuda_fifo_stream_init(kaapi_cuda_fifo_stream_t* fifo)
{
  const cudaError_t res = cudaStreamCreate(&fifo->stream);
  if( res != cudaSuccess ){
    fprintf( stdout, "%s: cudaStreamCreate ERROR %d\n", __FUNCTION__,
	    res );
    fflush(stdout);
    abort();
  }

  fifo->head = NULL;
  fifo->tail = NULL;
  fifo->cnt = 0;

  return 0;
}


/* Destroy a CUDA stream.
   \retval 0 in case of success
   \retval else an error code
*/
static inline int kaapi_cuda_fifo_stream_destroy(kaapi_cuda_fifo_stream_t* fifo)
{
  if (fifo->head !=0) return EINVAL;
  fifo->head = 0;
  fifo->tail = 0;

  cudaStreamDestroy(fifo->stream);

  return 0;
}

static inline kaapi_cuda_request_t* kaapi_cuda_fifo_stream_first( 
  kaapi_cuda_fifo_stream_t* fifo
)
{ /* first in fifo order: tail, because head insertion */
  return fifo->head;
}

/* Pop the top request return it
*/
static kaapi_cuda_request_t* kaapi_cuda_fifo_stream_pop(
  kaapi_cuda_fifo_stream_t* fifo
)
{
  kaapi_cuda_request_t* retval = fifo->head;
  if( retval == NULL )
      return NULL;

  fifo->head = retval->next;
  if (fifo->head == 0) 
    fifo->tail = 0;
  retval->next = 0;
  fifo->cnt--;

  /* destroy the cuda ressource. Even if the request is not complet,
     remove it from the queue and destroy the cuda event (if configured with).
  */
  kaapi_cuda_request_destroy( retval );
  
  return retval;
}



/* Here stream and current device must match
*/
static kaapi_cuda_request_t* kaapi_cuda_stream_request_create(
  kaapi_cuda_stream_t* stream,
  int                (*cbk)(),
  void*                arg1,
  void*                arg2
)
{
#if CONFIG_USE_EVENT
  cudaError_t res;
#endif

  /* todo: dynamic allocation of bloc of requests and put them into
     the free list
  */
  kaapi_cuda_request_t* req = stream->lfree;
  if (req ==0) return 0;


#if CONFIG_USE_EVENT
  res = cudaEventCreateWithFlags(&req->event, cudaEventDisableTiming);
  if( res != cudaSuccess ) {
      fprintf( stdout, "%s: cudaEventCreateWithFlags ERROR %d\n",
	      __FUNCTION__, res );
      fflush(stdout);
      abort();
  }
#endif

  /* unlink */
  stream->lfree   = req->next;
  
  /* init */
  req->status.state = KAAPI_CUDA_REQUEST_INIT;
  req->status.error = 0;
  req->u_fnc        = cbk;
  req->u_arg[0]     = arg1;
  req->u_arg[1]     = arg2;
  req->next         = 0;

  return req;
}


/* Destroy
*/
static int kaapi_cuda_stream_request_free(
  kaapi_cuda_stream_t* stream,
  kaapi_cuda_request_t* request
)
{
  request->next = stream->lfree;
  stream->lfree = request;

  return 0;
}


/* 
*/
int kaapi_cuda_stream_init(
	unsigned int capacity, 
	kaapi_cuda_proc_t* proc
    )
{
    unsigned int i;
    kaapi_cuda_stream_t* kstream = (kaapi_cuda_stream_t*)
	malloc( sizeof(kaapi_cuda_stream_t) );
    if( NULL == kstream ) return ENOMEM;
    proc->kstream = kstream;

    kstream->nodes = (kaapi_cuda_request_t*)
	malloc( capacity * sizeof(kaapi_cuda_request_t) );
    if (NULL == kstream->nodes) return ENOMEM;

    kstream->context = proc;

    /* initialize fifo_stream */
    if( kaapi_cuda_fifo_stream_init(&kstream->input_fifo) )
        goto on_error;

    if( kaapi_cuda_fifo_stream_init(&kstream->output_fifo) )
        goto on_error;

    if (kaapi_cuda_fifo_stream_init( &kstream->kernel_fifo) )
        goto on_error;

    /* form free list */
    kstream->lfree = &kstream->nodes[0];
    for (i=0; i < capacity-1; ++i)
        kstream->nodes[i].next = &kstream->nodes[i+1];
    kstream->nodes[capacity-1].next = NULL;

    return 0;
on_error:
    free(kstream);
    abort();
    return 0;
}


/*
*/
static inline kaapi_cuda_fifo_stream_t* get_kernel_fifo(kaapi_cuda_stream_t* stream)
{
#if CONFIG_USE_CONCURRENT_KERNELS
  /* round robin allocator */
  kaapi_cuda_fifo_stream_t* const fifo = &stream->kernel_fifos[stream->kernel_fifo_pos];
  stream->kernel_fifo_pos = (stream->kernel_fifo_pos + 1) % stream->kernel_fifo_count;
  return fifo;
#else
  return &stream->kernel_fifo;
#endif
}


/* exported version
*/
kaapi_cuda_fifo_stream_t* kaapi_cuda_get_kernel_fifo(kaapi_cuda_stream_t* stream)
{
  return get_kernel_fifo(stream);
}


/*
*/
static kaapi_cuda_request_t*
kaapi_cuda_fifo_stream_enqueue(
    kaapi_cuda_fifo_stream_t* fifostream,
    kaapi_cuda_request_t*     req
)
{
  cudaError_t err;
    
  /* insert in the tail of the queue */
  req->next = 0;
  if (fifostream->head == 0)
    fifostream->head = req;
  else 
    fifostream->tail->next = req;
  fifostream->tail = req;

#if CONFIG_USE_EVENT
  err = cudaEventRecord(req->event, fifostream->stream);
  if( err != cudaSuccess ){
    fprintf( stdout, "%s: cudaEventRecord ERROR %d\n", __FUNCTION__, err);
    fflush(stdout);
    cudaEventDestroy(req->event);
    abort();
  }
#endif
  req->status.state = KAAPI_CUDA_REQUEST_PUSH; 
  fifostream->cnt++;

  return req;
}


/* exported function
*/
kaapi_cuda_request_t* kaapi_cuda_stream_push1(
  kaapi_cuda_stream_t*    stream,
  kaapi_cuda_stream_op_t  op,
  ...
)
{
  kaapi_cuda_fifo_stream_t* fifostream;
  kaapi_cuda_request_t*     req;
  int                     (*cbk)();
  void*                     arg;
  va_list va_args;
  va_start(va_args, op);

  switch (op) {
    case KAAPI_CUDA_OP_H2D:
      fifostream = &stream->input_fifo;
      break;
    case KAAPI_CUDA_OP_D2H:
      fifostream = &stream->output_fifo;
      break;
    case KAAPI_CUDA_OP_KER:
      fifostream = get_kernel_fifo(stream);
      break;
    default:
      return 0;
  }
  cbk = va_arg(va_args, int (*)());
  arg = va_arg(va_args, void* );
  va_end(va_args);
  

  /* cread and push */    
  req = kaapi_cuda_stream_request_create( stream, cbk, arg, 0 );
  if (req == 0) 
  {
    kaapi_assert_debug( req != 0);
    return 0;
  }
  
  req = kaapi_cuda_fifo_stream_enqueue(fifostream, req );

  kaapi_assert_debug( req != 0);
  return 0;  
}



/* exported function
*/
kaapi_cuda_request_t* kaapi_cuda_stream_push2(
  kaapi_cuda_stream_t*    stream,
  kaapi_cuda_stream_op_t  op,
  ...
)
{
  kaapi_cuda_fifo_stream_t* fifostream;
  kaapi_cuda_request_t*     req;
  int                     (*cbk)();
  void*                     arg1;
  void*                     arg2;
  va_list va_args;
  va_start(va_args, op);

  switch (op) {
    case KAAPI_CUDA_OP_H2D:
      fifostream = &stream->input_fifo;
      break;
    case KAAPI_CUDA_OP_D2H:
      fifostream = &stream->output_fifo;
      break;
    case KAAPI_CUDA_OP_KER:
      fifostream = get_kernel_fifo(stream);
      break;
    default:
      return 0;
  }
  cbk = va_arg(va_args, int (*)());
  arg1 = va_arg(va_args, void* );
  arg2 = va_arg(va_args, void* );
  va_end(va_args);
  

  /* cread and push */    
  req = kaapi_cuda_stream_request_create( stream, cbk, arg1, arg2 );
  if (req == 0) 
  {
    kaapi_assert_debug( req != 0);
    return 0;
  }
  
  req = kaapi_cuda_fifo_stream_enqueue(fifostream, req );

  kaapi_assert_debug( req != 0);
  return 0;  
}


/* Req is the request which success cuEventQuery or cuStreamQuery.
   Signal all request between the first request and the request last
   (included).
   Previous pushed request are freed.
*/
static void kaapi_cuda_fifo_stream_signalall( 
  kaapi_cuda_stream_t*      stream,
  kaapi_cuda_fifo_stream_t* fifostream,
  kaapi_cuda_request_t*     last
)
{ 
  kaapi_cuda_request_t* req;
  
  /* because stream are fifo -> all previous requests are ready */
  while ( (req = kaapi_cuda_fifo_stream_pop( fifostream )) !=0 )
  {
    if (req->u_fnc !=0)
      req->status.error = req->u_fnc( stream, req->u_arg[0], req->u_arg[1] );
    kaapi_cuda_stream_request_free( stream, req );
    if (req == last)
      break;
  }
}


/** Wait end of the fifo stream.
    Return ENOENT if the stream is empty
*/
static int kaapi_cuda_wait_fifo_stream(
  kaapi_cuda_stream_t*      stream,
  kaapi_cuda_fifo_stream_t* fifostream
)
{
  cudaError_t err;
  
  if (fifostream->head ==0) 
    return 0;
    
  err = cudaStreamSynchronize( fifostream->stream );
  if ( err == cudaSuccess ) {
    /* here: all previously pushed event have completed 
       thus signal all posted request from the begin until the tail of the list.
    */
    kaapi_cuda_fifo_stream_signalall( stream, fifostream, fifostream->tail );
    return 0;
  }
#if defined(KAAPI_DEBUG)
  else {
      fprintf( stdout, "%s: cudaStreamSynchronize ERROR %d\n",
	      __FUNCTION__, err );
      fflush( stdout );
      abort();
  }
#endif
  return 1;
}


/** Wait the completion of all requests on the fifo stream
*/
extern kaapi_cuda_stream_state_t kaapi_cuda_wait_stream(
  kaapi_cuda_stream_t*    stream
)
{
  kaapi_cuda_wait_fifo_stream( stream, &stream->input_fifo );
  kaapi_cuda_wait_fifo_stream( stream, &stream->output_fifo );
  kaapi_cuda_wait_fifo_stream( stream, &stream->kernel_fifo );

  return KAAPI_CUDA_STREAM_READY;
}

static inline kaapi_cuda_stream_state_t
kaapi_cuda_testfirst_fifo_stream(
	kaapi_cuda_stream_t*      stream,
	kaapi_cuda_fifo_stream_t* fifostream
)
{
    cudaError_t err;
    kaapi_cuda_request_t* first = kaapi_cuda_fifo_stream_first(fifostream);

    if( NULL == first ) return KAAPI_CUDA_STREAM_EMPTY;

    err = cudaEventQuery( first->event );
    if( err == cudaSuccess ) {
	kaapi_cuda_fifo_stream_signalall( stream, fifostream, first );
	return KAAPI_CUDA_STREAM_READY;
    }
#if defined(KAAPI_DEBUG)
    else if( err != cudaErrorNotReady ) {
	fprintf(stdout, "%s: cudaEventQuery ERROR %d\n",
	__FUNCTION__, err );
	fflush(stdout);
	abort();
    }
#endif
    return KAAPI_CUDA_STREAM_BUSY;
}


/**
*/
static inline kaapi_cuda_stream_state_t
kaapi_cuda_test_fifo_stream(
	kaapi_cuda_stream_t*      stream,
	kaapi_cuda_fifo_stream_t* fifostream
)
{
  cudaError_t err;
  kaapi_cuda_request_t* first = kaapi_cuda_fifo_stream_first(fifostream);
  
  if( NULL == first ) return KAAPI_CUDA_STREAM_EMPTY;

  while( NULL != first ) {
    err = cudaEventQuery( first->event );
    if( err == cudaSuccess ) {
      kaapi_cuda_fifo_stream_signalall( stream, fifostream, first );
      return KAAPI_CUDA_STREAM_READY;
    } else if( err == cudaErrorNotReady )
	break;
#if defined(KAAPI_DEBUG)
    if( err != cudaErrorNotReady ) {
	fprintf(stdout, "%s: cudaEventQuery ERROR %d\n",
		__FUNCTION__, err );
	fflush(stdout);
	abort();
    }
#endif
    first = first->next;
  }
  return KAAPI_CUDA_STREAM_BUSY;
}


/** Test the completion of some requests on the fifo stream
*/
kaapi_cuda_stream_state_t
kaapi_cuda_test_stream(
	kaapi_cuda_stream_t*    stream
)
{
  kaapi_cuda_test_fifo_stream( stream, &stream->input_fifo );
  kaapi_cuda_test_fifo_stream( stream, &stream->kernel_fifo );
  kaapi_cuda_test_fifo_stream( stream, &stream->output_fifo );

  return KAAPI_CUDA_STREAM_READY;
}


/**
*/
static inline kaapi_cuda_stream_state_t
kaapi_cuda_waitsome_fifo_stream(
	kaapi_cuda_stream_t*      stream,
	kaapi_cuda_fifo_stream_t* fifostream
)
{
#if CONFIG_USE_EVENT
  /* test all requests begining from the oldest (the first to complet) */
  cudaError_t err;
  kaapi_cuda_request_t* first = kaapi_cuda_fifo_stream_first(fifostream);
  
  if( NULL == first ) return KAAPI_CUDA_STREAM_EMPTY;

  while( NULL != first ) {
    err = cudaEventQuery(first->event);
    if( err == cudaSuccess ) {
      kaapi_cuda_fifo_stream_signalall( stream, fifostream, first );
      return KAAPI_CUDA_STREAM_READY;
    }
#if defined(KAAPI_DEBUG)
    if( err != cudaErrorNotReady ) {
	fprintf(stdout, "%s: cudaEventQuery ERROR %d\n",
		__FUNCTION__, err );
	fflush(stdout);
	abort();
    }
#endif
    first = first->next;
  }

  return KAAPI_CUDA_STREAM_BUSY;

#else  // #if CONFIG_USE_EVENT
  /* no event: wait on all the stream */
  kaapi_cuda_wait_fifo_stream(fifostream);
  return 0;
#endif
}


/** Wait the completion of all requests on the fifo stream
*/
extern kaapi_cuda_stream_state_t
kaapi_cuda_waitsome_stream(
    kaapi_cuda_stream_t*    stream
)
{
  kaapi_cuda_waitsome_fifo_stream( stream, &stream->input_fifo );
  kaapi_cuda_waitsome_fifo_stream( stream, &stream->output_fifo );
  kaapi_cuda_waitsome_fifo_stream( stream, &stream->kernel_fifo );

  return 0;
}

static inline kaapi_cuda_stream_state_t
kaapi_cuda_waitfirst_fifo_stream(
	kaapi_cuda_stream_t*      stream,
	kaapi_cuda_fifo_stream_t* fifostream
)
{
#if CONFIG_USE_EVENT
  /* test all requests begining from the oldest (the first to complet) */
  cudaError_t err;
//  kaapi_cuda_request_t* first = fifostream->head;
  kaapi_cuda_request_t* first = kaapi_cuda_fifo_stream_first(fifostream);
  if( first == NULL ) return KAAPI_CUDA_STREAM_EMPTY;
  
  err = cudaEventSynchronize(first->event);
  if( err == cudaSuccess ) {
    kaapi_cuda_fifo_stream_signalall( stream, fifostream, first );
    return KAAPI_CUDA_STREAM_READY;
  }
#if defined(KAAPI_DEBUG)
  if( err != cudaErrorNotReady ){
      fprintf( stdout, "%s: cudaEventSynchronize ERROR %d\n",
	      __FUNCTION__, err );
      fflush( stdout );
      abort();
  }
#endif
  return KAAPI_CUDA_STREAM_BUSY;

#else  // #if CONFIG_USE_EVENT
  /* no event: wait on all the stream */
  kaapi_cuda_wait_fifo_stream(fifostream);
  return 0;
#endif
}


/** Wait the completion of the first requests in the fifo stream.
    Priority:
      - input
      - output
      - kernel
    Once a stream contains a request, wait onto the request and return
    without testing next fifo streams.
TODO: cudaEventSynchronize synchronizes GPU.
*/
kaapi_cuda_stream_state_t
kaapi_cuda_waitfirst_stream(
        kaapi_cuda_stream_t*    stream
)
{
    kaapi_cuda_stream_state_t st[3];
    while( 1 ){
	st[0]= kaapi_cuda_testfirst_fifo_stream( stream, &stream->input_fifo );
	st[1]= kaapi_cuda_testfirst_fifo_stream( stream, &stream->output_fifo );
	st[2]= kaapi_cuda_testfirst_fifo_stream( stream, &stream->kernel_fifo );
	if( (st[0] == KAAPI_CUDA_STREAM_READY) ||
		(st[1] == KAAPI_CUDA_STREAM_READY) ||
		(st[2] == KAAPI_CUDA_STREAM_READY) )
	    return KAAPI_CUDA_STREAM_READY;

	if( (st[0] == KAAPI_CUDA_STREAM_EMPTY) &&
		(st[1] == KAAPI_CUDA_STREAM_EMPTY) &&
		(st[2] == KAAPI_CUDA_STREAM_EMPTY) )
	    return KAAPI_CUDA_STREAM_EMPTY;
	    
    }
    return KAAPI_CUDA_STREAM_READY;
}


void kaapi_cuda_stream_destroy( kaapi_cuda_stream_t* stream )
{
    kaapi_cuda_fifo_stream_destroy( &stream->input_fifo );
    kaapi_cuda_fifo_stream_destroy( &stream->output_fifo );
    kaapi_cuda_fifo_stream_destroy( &stream->input_fifo );
#if CONFIG_USE_CONCURRENT_KERNELS
    for (i = 0; i < stream->kernel_fifo_count; ++i)
	kaapi_cuda_fifo_stream_destroy( &stream->kernel_fifos[i] );
#else /* ! CONFIG_USE_CONCURRENT_KERNELS */
    kaapi_cuda_fifo_stream_destroy( &stream->kernel_fifo );
#endif /* CONFIG_USE_CONCURRENT_KERNELS */
}

kaapi_cuda_stream_state_t
kaapi_cuda_waitfirst_input( kaapi_cuda_stream_t*      stream )
{
    return kaapi_cuda_waitfirst_fifo_stream( stream, kaapi_cuda_get_input_fifo(stream) );
}

kaapi_cuda_stream_state_t
kaapi_cuda_waitfirst_kernel( kaapi_cuda_stream_t*      stream )
{
    return kaapi_cuda_waitfirst_fifo_stream( stream, kaapi_cuda_get_kernel_fifo(stream) );
}

void
kaapi_cuda_stream_poll( kaapi_processor_t* const kproc )
{
    kaapi_cuda_stream_t* const kstream = kproc->cuda_proc.kstream;
    if( !kaapi_cuda_stream_is_empty( kstream ) )
	kaapi_cuda_test_stream( kstream );
}
