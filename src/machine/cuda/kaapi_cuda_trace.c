/*
** kaapi_cuda_data_async.h
** xkaapi
** 
** Created on Jan 2012
** Copyright 2010 INRIA.
**
** Contributors :
**
** thierry.gautier@inrialpes.fr
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

#if defined(KAAPI_USE_PERFCOUNTER) && defined(KAAPI_USE_CUPTI)

#include <stdio.h>
#include <cuda_runtime_api.h>
#include <cupti.h>

#include "kaapi_impl.h"
#include "kaapi_event.h"
#include "kaapi_cuda_event.h"

#define ALIGN_SIZE (8)
#define BUFFER_SIZE	(32 * 1024)
#define ALIGN_BUFFER(buffer, align) \
  (((uintptr_t) (buffer) & ((align)-1)) ? ((buffer) + (align) - ((uintptr_t) (buffer) & ((align)-1))) : (buffer)) 

#if 0
static const char *
getMemcpyKindString(CUpti_ActivityMemcpyKind kind)
{
  switch (kind) {
  case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD:
    return "HtoD";
  case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH:
    return "DtoH";
  case CUPTI_ACTIVITY_MEMCPY_KIND_HTOA:
    return "HtoA";
  case CUPTI_ACTIVITY_MEMCPY_KIND_ATOH:
    return "AtoH";
  case CUPTI_ACTIVITY_MEMCPY_KIND_ATOA:
    return "AtoA";
  case CUPTI_ACTIVITY_MEMCPY_KIND_ATOD:
    return "AtoD";
  case CUPTI_ACTIVITY_MEMCPY_KIND_DTOA:
    return "DtoA";
  case CUPTI_ACTIVITY_MEMCPY_KIND_DTOD:
    return "DtoD";
  case CUPTI_ACTIVITY_MEMCPY_KIND_HTOH:
    return "HtoH";
  default:
    break;
  }

  return "<unknown>";
}
#endif

static inline void
kaapi_cuda_trace_record_memcpy( kaapi_processor_t *kproc,
	CUpti_ActivityMemcpy* memcpy  )
{
    CUpti_ActivityMemcpyKind kind =
	(CUpti_ActivityMemcpyKind)memcpy->copyKind;

    switch (kind) {
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD:
	KAAPI_CUDA_EVENT_PUSH0_( kproc, memcpy->start, NULL, KAAPI_EVT_CUDA_GPU_HTOD_BEG );
	KAAPI_CUDA_EVENT_PUSH0_( kproc, memcpy->end, NULL, KAAPI_EVT_CUDA_GPU_HTOD_END );
	break;
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH:
	KAAPI_CUDA_EVENT_PUSH0_( kproc, memcpy->start, NULL, KAAPI_EVT_CUDA_GPU_DTOH_BEG );
	KAAPI_CUDA_EVENT_PUSH0_( kproc, memcpy->end, NULL, KAAPI_EVT_CUDA_GPU_DTOH_END );
	break;
    default:
	break;
    }
}

static inline void
kaapi_cuda_trace_record_kernel( kaapi_processor_t *kproc,
	CUpti_ActivityKernel* kernel  )
{
    KAAPI_CUDA_EVENT_PUSH0_( kproc, kernel->start, NULL, KAAPI_EVT_CUDA_GPU_KERNEL_BEG );
    KAAPI_CUDA_EVENT_PUSH0_( kproc, kernel->end, NULL, KAAPI_EVT_CUDA_GPU_KERNEL_END );
}

static inline void
kaapi_cuda_trace_record( CUpti_Activity *record )
{
    kaapi_processor_t* kproc;

    switch (record->kind) {
    case CUPTI_ACTIVITY_KIND_MEMCPY:
    {
	CUpti_ActivityMemcpy *memcpy = (CUpti_ActivityMemcpy *)record;
	kproc = kaapi_cuda_get_proc_by_dev( memcpy->deviceId );
	kaapi_cuda_trace_record_memcpy( kproc, memcpy );
#if 0
	printf("MEMCPY %s [ %llu - %llu ] device %u, context %u, stream %u, correlation %u/r%u\n",
	getMemcpyKindString((CUpti_ActivityMemcpyKind)memcpy->copyKind),
	(unsigned long long)(memcpy->start - kaapi_default_param.cudastartuptime),
	(unsigned long long)(memcpy->end - kaapi_default_param.cudastartuptime),
	memcpy->deviceId, memcpy->contextId, memcpy->streamId, 
	memcpy->correlationId, memcpy->runtimeCorrelationId);
#endif
	break;
    }
    case CUPTI_ACTIVITY_KIND_KERNEL:
    {
	CUpti_ActivityKernel *kernel = (CUpti_ActivityKernel *)record;
	kproc = kaapi_cuda_get_proc_by_dev( kernel->deviceId );
	kaapi_cuda_trace_record_kernel( kproc, kernel );
#if 0
	printf("KERNEL \"%s\" [ %llu - %llu ] device %u, context %u, stream %u, correlation %u/r%u\n",
	    kernel->name,
	    (unsigned long long)(kernel->start - kaapi_default_param.cudastartuptime),
	    (unsigned long long)(kernel->end - kaapi_default_param.cudastartuptime),
	    kernel->deviceId, kernel->contextId, kernel->streamId, 
	    kernel->correlationId, kernel->runtimeCorrelationId);
	printf("    grid [%u,%u,%u], block [%u,%u,%u], shared memory (static %u, dynamic %u)\n",
	kernel->gridX, kernel->gridY, kernel->gridZ,
	kernel->blockX, kernel->blockY, kernel->blockZ,
	kernel->staticSharedMemory, kernel->dynamicSharedMemory);
#endif
	break;
    }
    default:
	break;
    }
}


static inline void
kaapi_cuda_trace_add_buffer( CUcontext ctx, uint32_t stream )
{
    size_t size = BUFFER_SIZE;
    uint8_t* buffer = (uint8_t*) malloc( size+ALIGN_SIZE );
    const CUptiResult res = cuptiActivityEnqueueBuffer( ctx, stream, 
	    ALIGN_BUFFER(buffer, ALIGN_SIZE), size );
    if( res != CUPTI_SUCCESS ) {
	fprintf(stdout, "%s: cuptiActivityEnqueueBuffer ERROR %d\n", __FUNCTION__, res );
	fflush(stdout); 
    }
}


static uint8_t *
kaapi_cuda_trace_dump( CUcontext context, uint32_t streamId )
{
    uint8_t *buffer = NULL;
    size_t validBufferSizeBytes;
    CUptiResult status;
    status = cuptiActivityDequeueBuffer(context, streamId, &buffer,
	    &validBufferSizeBytes);
    if (status == CUPTI_ERROR_QUEUE_EMPTY)
	return NULL;

    if (context == NULL) {
    printf("==== Starting dump for global ====\n");
    } else if (streamId == 0) {
    printf("==== Starting dump for context %p ====\n", context);
    } else {
    printf("==== Starting dump for context %p, stream %u ====\n", context, streamId);
    }

    CUpti_Activity *record = NULL;
    do {
	status = cuptiActivityGetNextRecord(buffer, validBufferSizeBytes,
		&record);
	if(status == CUPTI_SUCCESS) {
	  kaapi_cuda_trace_record( record );
	}
	else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED) {
	  break;
	}
    } while (1);

    // report any records dropped from the queue
    size_t dropped;
    cuptiActivityGetNumDroppedRecords(context, streamId, &dropped);
    if (dropped != 0) 
	printf("# ERROR Dropped %u activity records\n", (unsigned int)dropped);

    return buffer;
}

static inline uint8_t *
kaapi_cuda_trace_record_if_full(CUcontext context, uint32_t streamId)
{
    size_t validBufferSizeBytes;
    CUptiResult status;
    status = cuptiActivityQueryBuffer(context, streamId, &validBufferSizeBytes);
    if (status == CUPTI_ERROR_MAX_LIMIT_REACHED)
	return kaapi_cuda_trace_dump(context, streamId);

    return NULL;
}

static void
kaapi_cuda_trace_handle( CUpti_CallbackId cbid,
	const CUpti_SynchronizeData *syncData )
{
    // check the top buffer of the global queue and dequeue if full. If
    // we dump a buffer add it back to the queue
    uint8_t *buffer = kaapi_cuda_trace_record_if_full(NULL, 0);
    if (buffer != NULL) {
	cuptiActivityEnqueueBuffer(NULL, 0, buffer, BUFFER_SIZE);
    }

    // dump context buffer on context sync
    if (cbid == CUPTI_CBID_SYNCHRONIZE_CONTEXT_SYNCHRONIZED) {
	buffer = kaapi_cuda_trace_record_if_full(syncData->context, 0);
	if (buffer != NULL) {
	  cuptiActivityEnqueueBuffer(syncData->context, 0, buffer, BUFFER_SIZE);
	}
    }
    // dump stream buffer on stream sync
    else if (cbid == CUPTI_CBID_SYNCHRONIZE_STREAM_SYNCHRONIZED) {
	uint32_t streamId;
	cuptiGetStreamId(syncData->context, syncData->stream, &streamId);
	buffer = kaapi_cuda_trace_record_if_full(syncData->context, streamId);
	if (buffer != NULL) {
	  cuptiActivityEnqueueBuffer(syncData->context, streamId, buffer,
		  BUFFER_SIZE);
	}
    }
}

static void CUPTIAPI
kaapi_cuda_trace_callback( void *userdata, CUpti_CallbackDomain domain,
              CUpti_CallbackId cbid, const void *cbdata )
{
#if 0
  if (domain == CUPTI_CB_DOMAIN_RESOURCE) {
    handleResource(cbid, (CUpti_ResourceData *)cbdata);
  } else
#endif
    if (domain == CUPTI_CB_DOMAIN_SYNCHRONIZE) {
	kaapi_cuda_trace_handle( cbid, (CUpti_SynchronizeData *)cbdata );
    }
}

int kaapi_cuda_trace_init( void )
{
    CUptiResult res;
    CUpti_SubscriberHandle subscriber;

    /* library init */
    kaapi_cuda_trace_add_buffer( NULL, 0 );

    res = cuptiActivityEnable( CUPTI_ACTIVITY_KIND_DEVICE );
    if( res != CUPTI_SUCCESS ) {
	fprintf(stdout, "%s: cuptiActivityEnable ERROR %d\n", __FUNCTION__, res );
	fflush(stdout); 
    }

    res = cuptiSubscribe( &subscriber,
	    (CUpti_CallbackFunc)kaapi_cuda_trace_callback, NULL);
    if( res != CUPTI_SUCCESS ) {
	fprintf(stdout, "%s: cuptiSubscribe ERROR %d\n", __FUNCTION__, res );
	fflush(stdout); 
    }

    res = cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_SYNCHRONIZE);
    if( res != CUPTI_SUCCESS ) {
	fprintf(stdout, "%s: cuptiEnableDomain ERROR %d\n", __FUNCTION__, res );
	fflush(stdout); 
    }
//    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONTEXT);
//    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER);
//    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME);
//    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET);
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY);
    cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL);
//
    cuptiGetTimestamp( &kaapi_default_param.cudastartuptime );

    return 0;
}

int kaapi_cuda_trace_thread_init( void )
{
    CUptiResult res;
    CUcontext ctx;
    uint32_t stream_id;

    cuCtxGetCurrent(&ctx);
    res = cuptiGetStreamId( ctx, (CUstream)kaapi_cuda_HtoD_stream(),
	    &stream_id );
    if( res != CUPTI_SUCCESS ) {
	fprintf(stdout, "%s: cuptiGetStreamId ERROR %d\n", __FUNCTION__, res );
	fflush(stdout); 
    }
    kaapi_cuda_trace_add_buffer( ctx,  stream_id );

    res = cuptiGetStreamId( ctx, (CUstream)kaapi_cuda_kernel_stream(),
	    &stream_id );
    if( res != CUPTI_SUCCESS ) {
	fprintf(stdout, "%s: cuptiGetStreamId ERROR %d\n", __FUNCTION__, res );
	fflush(stdout); 
    }
    kaapi_cuda_trace_add_buffer( ctx,  stream_id );

    return 0;
}

void kaapi_cuda_trace_finalize( void )
{
   while( kaapi_cuda_trace_dump( NULL, 0 ) != NULL ) ;
}

void kaapi_cuda_trace_thread_finalize( void )
{
    CUptiResult res;
    CUcontext ctx;
    uint32_t stream_id;

    cuCtxGetCurrent(&ctx);
    res = cuptiGetStreamId( ctx, (CUstream)kaapi_cuda_HtoD_stream(),
	    &stream_id );
    if( res != CUPTI_SUCCESS ) {
	fprintf(stdout, "%s: cuptiGetStreamId ERROR %d\n", __FUNCTION__, res );
	fflush(stdout); 
    }
    while( kaapi_cuda_trace_dump( ctx, stream_id ) != NULL ) ;

    res = cuptiGetStreamId( ctx, (CUstream)kaapi_cuda_kernel_stream(),
	    &stream_id );
    if( res != CUPTI_SUCCESS ) {
	fprintf(stdout, "%s: cuptiGetStreamId ERROR %d\n", __FUNCTION__, res );
	fflush(stdout); 
    }
    while( kaapi_cuda_trace_dump( ctx, stream_id ) != NULL ) ;

    res = cuptiGetStreamId( ctx, (CUstream)kaapi_cuda_DtoH_stream(),
	    &stream_id );
    if( res != CUPTI_SUCCESS ) {
	fprintf(stdout, "%s: cuptiGetStreamId ERROR %d\n", __FUNCTION__, res );
	fflush(stdout); 
    }
    while( kaapi_cuda_trace_dump( ctx, stream_id ) != NULL ) ;
}

#endif

