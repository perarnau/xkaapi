
#ifndef KAAPI_CUDA_MEM_H_INCLUDED
#define KAAPI_CUDA_MEM_H_INCLUDED

#include "../../kaapi.h"
#include "kaapi_cuda_proc.h"

#define KAAPI_CUDA_ASYNC	1

#define KAAPI_CUDA_MEM_FREE_FACTOR  1

int kaapi_cuda_mem_free( kaapi_pointer_t *ptr );

int kaapi_cuda_mem_alloc(
		kaapi_metadata_info_t*  mdi,
	       	kaapi_pointer_t *ptr,
		const kaapi_address_space_id_t kasid,
		const size_t size, const int flag );

int kaapi_cuda_mem_register( kaapi_pointer_t ptr, 
		const kaapi_memory_view_t *view );

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

#endif /* ! KAAPI_CUDA_MEM_H_INCLUDED */
