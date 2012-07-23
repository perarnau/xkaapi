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
** Joao.Lima@imag.fr
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
#ifndef KAAPI_CUDA_STREAM_H_INCLUDED
#define KAAPI_CUDA_STREAM_H_INCLUDED

#include <stdint.h>
#include <stdlib.h>
#include <stddef.h>
#include <sys/types.h>

struct kaapi_cuda_fifo_stream_t;
struct kaapi_cuda_stream_t;

#define CONFIG_USE_EVENT 1

typedef enum {
  KAAPI_CUDA_REQUEST_INIT,
  KAAPI_CUDA_REQUEST_PUSH,
  KAAPI_CUDA_REQUEST_TERM,
  KAAPI_CUDA_REQUEST_ERROR
} kaapi_cuda_request_state_t;

typedef enum {
  KAAPI_CUDA_STREAM_EMPTY,
  KAAPI_CUDA_STREAM_READY,
  KAAPI_CUDA_STREAM_BUSY
} kaapi_cuda_stream_state_t;

/*
*/
typedef struct kaapi_cuda_status_t {
  int state;
  int error;
} kaapi_cuda_status_t;

typedef int (*kaapi_cuda_stream_callback_t) (struct kaapi_cuda_stream_t *,
					     void *);

typedef struct kaapi_cuda_request_t {
  kaapi_cuda_status_t status;
  kaapi_cuda_stream_callback_t fnc;
  void *arg;
#if CONFIG_USE_EVENT
  cudaEvent_t event;
#endif
  struct kaapi_cuda_request_t *next;	/* next following fifo order */
} kaapi_cuda_request_t;

/* This is the Kaapi view of a Cuda stream.
   A Kaapi Cuda stream allows the user to insert asynchronous
   operations that complete in a Fifo order.
*/
typedef struct kaapi_cuda_fifo_stream_t {
  cudaStream_t stream;
  uint64_t cnt;			/* number of requests */
  kaapi_cuda_request_t *head;	/* first in the fifo order (insertion in tail) */
  kaapi_cuda_request_t *tail;	/* last pushed in the fifo order */
} kaapi_cuda_fifo_stream_t;


//#define CONFIG_USE_CONCURRENT_KERNELS 1

/* Kaapi CUDA stream allows to insert asynchronous
   memory operation (H2D or D2H) and asynchronous
   kernel invocation.
*/
typedef struct kaapi_cuda_stream_t {
//  struct kaapi_cuda_proc_t* context;
  kaapi_cuda_proc_t *context;

  kaapi_cuda_fifo_stream_t input_fifo;
  kaapi_cuda_fifo_stream_t output_fifo;

#if CONFIG_USE_CONCURRENT_KERNELS
  /* round robin allocator */
  unsigned int kernel_fifo_pos;
  unsigned int kernel_fifo_count;
  kaapi_cuda_fifo_stream_t kernel_fifos[4];
#else
  kaapi_cuda_fifo_stream_t kernel_fifo;
#endif

  /* request allocator */
  kaapi_cuda_request_t *lfree;
  kaapi_cuda_request_t *nodes;

} kaapi_cuda_stream_t;


/* Create a kaapi_cuda_stream_t object.
   The routine allocates and initializes
   a kaapi cuda stream for attached to the cuda
   kprocessor proc.
   The capacity value is the capacity of the stream
   to handle pending asynchronous operation.
   If for some usage, the number of pending asynchronous
   operation is greather than this capacity, then
   the stream will do not accept new asynchronous operation
   until a previously pushed operation completes.
*/
extern int kaapi_cuda_stream_init(unsigned int capacity,
				  kaapi_cuda_proc_t * proc);

extern void kaapi_cuda_stream_destroy(kaapi_cuda_stream_t * stream);

typedef enum {
  KAAPI_CUDA_OP_KER = 0,	/* kernel launch */
  KAAPI_CUDA_OP_H2D = 1,	/* host 2 device operation */
  KAAPI_CUDA_OP_D2H = 2		/* device 2 host operation */
} kaapi_cuda_stream_op_t;


/* Push a new asynchronous event into the kaapi_cuda_stream.
   Depending of the kind of operation, the event is recorded
   into one of the different underlaying cuda stream.
   
   On the completion of the event, the runtime calls the call back
   function cbk(cu_stream, arg_action) and stores the return value 
   into the status of the request.
   The return value of the callback function is avaible in the
   request status, once the user has tested or wait for the handle.
   
   The call back function may be null.
   
   All pushed requests with the same type of operation are enqueued 
   in a fifo maner and they complet in order: the runtime invokes 
   the callback in the same order as the requests were pushed.
*/
extern struct kaapi_cuda_request_t
*kaapi_cuda_stream_push(kaapi_cuda_stream_t * const stream,
			const kaapi_cuda_stream_op_t op,
			kaapi_cuda_stream_callback_t fnc,
			void *const arg);

/** Blocking operation
*/
extern kaapi_cuda_stream_state_t kaapi_cuda_wait_stream(kaapi_cuda_stream_t
							* stream);

/** Wait the completion of the first requests in the fifo stream.
    Priority:
      - input
      - output
      - kernel
    Once a stream contains a request, wait onto the request and return
    without testing next fifo streams.
    
    \retval ENOENT if all streams are empty
*/
extern kaapi_cuda_stream_state_t
kaapi_cuda_waitfirst_stream(kaapi_cuda_stream_t * stream);

/** Non blocking operation
*/
extern kaapi_cuda_stream_state_t
kaapi_cuda_test_stream(kaapi_cuda_stream_t * stream);

/**
*/
extern kaapi_cuda_fifo_stream_t
    * kaapi_cuda_get_kernel_fifo(kaapi_cuda_stream_t * stream);

/**
*/
static inline kaapi_cuda_fifo_stream_t
    * kaapi_cuda_get_input_fifo(kaapi_cuda_stream_t * stream)
{
  return &stream->input_fifo;
}

/**
*/
static inline kaapi_cuda_fifo_stream_t
    * kaapi_cuda_get_output_fifo(kaapi_cuda_stream_t * stream)
{
  return &stream->output_fifo;
}

/**
*/
static inline cudaStream_t
kaapi_cuda_get_cudastream(kaapi_cuda_fifo_stream_t * stream)
{
  return stream->stream;
}

static inline uint64_t
kaapi_cuda_get_active_count_fifo(kaapi_cuda_fifo_stream_t * stream)
{
  return stream->cnt;
}

kaapi_cuda_stream_state_t
kaapi_cuda_waitfirst_input(kaapi_cuda_stream_t * stream);


kaapi_cuda_stream_state_t
kaapi_cuda_waitfirst_kernel(kaapi_cuda_stream_t * stream);

static inline int kaapi_cuda_stream_is_empty(kaapi_cuda_stream_t * kstream)
{
  return ((kaapi_cuda_get_active_count_fifo
	   (kaapi_cuda_get_input_fifo(kstream)) == 0)
	  &&
	  (kaapi_cuda_get_active_count_fifo
	   (kaapi_cuda_get_output_fifo(kstream)) == 0)
	  &&
	  (kaapi_cuda_get_active_count_fifo
	   (kaapi_cuda_get_kernel_fifo(kstream)) == 0));
}

void kaapi_cuda_stream_poll(kaapi_processor_t * const);

static inline void
kaapi_cuda_stream_window_test(kaapi_cuda_stream_t * kstream)
{
  kaapi_cuda_test_stream(kstream);
  /* The slicing window is applied to all streams */
  while ((kaapi_default_param.cudawindowsize <=
	  kaapi_cuda_get_active_count_fifo(kaapi_cuda_get_input_fifo
					   (kstream)))
	 || (kaapi_default_param.cudawindowsize <=
	     kaapi_cuda_get_active_count_fifo(kaapi_cuda_get_kernel_fifo
					      (kstream)))
	 || (kaapi_default_param.cudawindowsize <=
	     kaapi_cuda_get_active_count_fifo(kaapi_cuda_get_output_fifo
					      (kstream)))) {
    kaapi_cuda_test_stream(kstream);
  }
  kaapi_cuda_test_stream(kstream);
}

static inline void kaapi_cuda_stream_waitall(kaapi_cuda_stream_t * kstream)
{
  while (!kaapi_cuda_stream_is_empty(kstream))
    kaapi_cuda_test_stream(kstream);
}

#endif
