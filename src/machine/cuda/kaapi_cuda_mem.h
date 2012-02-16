
#ifndef KAAPI_CUDA_MEM_H_INCLUDED
#define KAAPI_CUDA_MEM_H_INCLUDED

#include "../../kaapi.h"
#include "kaapi_cuda_proc.h"
#include <cuda_runtime_api.h>

int kaapi_cuda_mem_free( kaapi_pointer_t *ptr );

int kaapi_cuda_mem_alloc(
	       	kaapi_pointer_t *ptr,
		const kaapi_address_space_id_t kasid,
		const size_t size, const int flag );

int kaapi_cuda_mem_register( kaapi_pointer_t ptr, 
		const kaapi_memory_view_t *view );

#ifdef	KAAPI_CUDA_MEM_ALLOC_MANAGER
int
kaapi_cuda_mem_inc_use( kaapi_pointer_t *ptr );
#endif

int kaapi_cuda_mem_copy_htod(
	kaapi_pointer_t dest, const kaapi_memory_view_t* view_dest,
	const kaapi_pointer_t src, const kaapi_memory_view_t* view_src
	       	);

int kaapi_cuda_mem_copy_dtoh(
	kaapi_pointer_t dest, const kaapi_memory_view_t* view_dest,
	const kaapi_pointer_t src, const kaapi_memory_view_t* view_src
		);

int kaapi_cuda_mem_1dcopy_htod(
	kaapi_pointer_t dest, const kaapi_memory_view_t* view_dest,
	const kaapi_pointer_t src, const kaapi_memory_view_t* view_src
	);

int kaapi_cuda_mem_1dcopy_dtoh(
	kaapi_pointer_t dest, const kaapi_memory_view_t* view_dest,
	const kaapi_pointer_t src, const kaapi_memory_view_t* view_src
	);

int kaapi_cuda_mem_2dcopy_htod(
	kaapi_pointer_t dest, const kaapi_memory_view_t* view_dest,
	const kaapi_pointer_t src, const kaapi_memory_view_t* view_src
	);

int kaapi_cuda_mem_2dcopy_dtoh( 
	kaapi_pointer_t dest, const kaapi_memory_view_t* view_dest,
	const kaapi_pointer_t src, const kaapi_memory_view_t* view_src
	);

int kaapi_cuda_mem_sync_params( 
	kaapi_thread_context_t* thread,
	kaapi_taskdescr_t*         td,
	kaapi_task_t*              pc
);

int kaapi_cuda_mem_sync_params_dtoh( 
	kaapi_thread_context_t* thread,
	kaapi_taskdescr_t*         td,
	kaapi_task_t*              pc
);

#if	KAAPI_CUDA_MEM_ALLOC_MANAGER
int 
kaapi_cuda_mem_mgmt_check( kaapi_processor_t* proc );
#endif

static inline int
kaapi_cuda_mem_register_( void* ptr, const size_t size )
{
#if KAAPI_CUDA_ASYNC
    const cudaError_t res= cudaHostRegister( ptr, size, cudaHostRegisterPortable );
#if 0
    if (res != cudaSuccess) {
	    fprintf( stdout, "[%s] ERROR (%d) ptr=%p size=%lu kid=%lu\n",
			    __FUNCTION__, res,
			    ptr, size,
			    (long unsigned int)kaapi_get_current_kid() ); 
	    fflush( stdout );
    }
#endif

    return res;
#else
    return 0;
#endif
}

static inline void
kaapi_cuda_mem_unregister_( void* ptr )
{
#if KAAPI_CUDA_ASYNC
    cudaHostUnregister( ptr );
#endif
}

#if KAAPI_CUDA_ASYNC
int kaapi_cuda_mem_copy_dtoh_(
	kaapi_pointer_t dest, const kaapi_memory_view_t* view_dest,
	const kaapi_pointer_t src, const kaapi_memory_view_t* view_src,
	cudaStream_t stream
		);

int kaapi_cuda_mem_1dcopy_dtoh_(
	kaapi_pointer_t dest, const kaapi_memory_view_t* view_dest,
	const kaapi_pointer_t src, const kaapi_memory_view_t* view_src,
	cudaStream_t stream
	);

int kaapi_cuda_mem_2dcopy_dtoh_( 
	kaapi_pointer_t dest, const kaapi_memory_view_t* view_dest,
	const kaapi_pointer_t src, const kaapi_memory_view_t* view_src,
	cudaStream_t stream
	);
#endif

#endif /* ! KAAPI_CUDA_MEM_H_INCLUDED */
