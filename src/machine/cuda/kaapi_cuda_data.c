
#include <cuda.h>

#include "kaapi_impl.h"
#include "../../kaapi_memory.h"
#include "kaapi_cuda_proc.h"
#include "kaapi_cuda_utils.h"
#include "kaapi_cuda_mem.h"
#include "kaapi_cuda_data.h"

/* here it checks if the CPU pointer is present in the GPU.
 * If not, it allocates.
 * Context: it executes right before a CUDA task (kernel).
 */
static inline int
xxx_kaapi_cuda_data_allocate(
		kaapi_thread_context_t* thread,
		kaapi_metadata_info_t*  mdi,
		kaapi_data_t* dest,
		kaapi_data_t* src
		)
{
	if( __kaapi_pointer2void(dest->ptr) == NULL ) {
		kaapi_cuda_mem_alloc( mdi, &dest->ptr, thread->asid,
			kaapi_memory_view_size(&dest->view), 0 );

		_kaapi_metadata_info_bind_data( mdi, thread->asid,
			__kaapi_pointer2void(dest->ptr), &dest->view );

//		kaapi_mem_findinsert_metadata2( __kaapi_pointer2void(dest->ptr), mdi );

		kaapi_cuda_mem_register( src->ptr, &src->view );

//		_kaapi_metadata_info_set_invalid( mdi, thread->asid );
//		kaapi_cuda_mem_copy_htod( dest->ptr, &dest->view, 
//			src->ptr, &src->view );
	}

	return 0;
}

/* The function checks if the dest memory is valid on GPU 
 * */
static inline int
xxx_kaapi_cuda_data_send_ro(
		kaapi_thread_context_t* thread,
		kaapi_metadata_info_t*  mdi,
		kaapi_data_t* dest,
		kaapi_data_t* src
		)
{
	if ( !_kaapi_metadata_info_is_valid(mdi, thread->asid) ) {
		kaapi_cuda_mem_copy_htod( dest->ptr, &dest->view,
			src->ptr, &src->view );
//		_kaapi_metadata_info_set_valid( mdi, thread->asid );
	}
	return 0;
}

/*
 * The function sets the current GPU thread as a writter 
 */
static inline int
xxx_kaapi_cuda_data_send_wr(
		kaapi_thread_context_t* thread,
		kaapi_metadata_info_t*  mdi,
		kaapi_data_t* dest, 
	       	kaapi_data_t* src
		)
{
	_kaapi_metadata_info_set_writer( mdi, thread->asid );
	return 0;
}

/* 
 * Context: it executes right before a CUDA task (kernel).
 * The function goes through every parameter and checks if it is allocated and
 * valid on GPU memory.
 */
int kaapi_cuda_data_send( 
	kaapi_thread_context_t* thread,
	kaapi_format_t*		   fmt,
	kaapi_task_t*              pc
)
{
	size_t count_params = kaapi_format_get_count_params(fmt, pc->sp );
	unsigned int i;
	kaapi_metadata_info_t*  mdi;
//	kaapi_mem_mapping_t* mapping;
//	kaapi_mem_map_t* const host_map = &kaapi_all_kprocessors[0]->mem_map;
//	kaapi_mem_map_t* const self_map = kaapi_get_current_mem_map();
//	kaapi_mem_asid_t asid = kaapi_mem_map_get_asid( self_map );

#if KAAPI_VERBOSE
	fprintf(stdout, "[%s] CUDA params=%ld kid=%lu\n", __FUNCTION__,
		count_params,
		(unsigned long)kaapi_get_current_kid() );
	fflush( stdout );
#endif
	for ( i=0; i < count_params; i++ ) {
		kaapi_access_mode_t m = KAAPI_ACCESS_GET_MODE(
			kaapi_format_get_mode_param( fmt, i, pc->sp) );
		if (m == KAAPI_ACCESS_MODE_V) 
			continue;

		kaapi_access_t access = kaapi_format_get_access_param( fmt,
			       	i, pc->sp );
		kaapi_data_t* src = (kaapi_data_t*)access.data;
		mdi = src->mdi;
//		mdi = kaapi_memory_find_metadata( __kaapi_pointer2void(src->ptr) );

		kaapi_data_t* dest = _kaapi_metadata_info_get_data( mdi, thread->asid );
		dest->view = kaapi_format_get_view_param( fmt,
			i, pc->sp );
		dest->mdi= mdi;

		xxx_kaapi_cuda_data_allocate( thread, mdi, dest, src );

		if( KAAPI_ACCESS_IS_READ(m) )
			xxx_kaapi_cuda_data_send_ro( thread, mdi, dest, src );

		if( KAAPI_ACCESS_IS_WRITE(m) )
			xxx_kaapi_cuda_data_send_wr( thread, mdi, dest, src );

		/* sets new pointer to the task */
		access.data =  dest;
		kaapi_format_set_access_param( fmt, i, pc->sp, &access );
	}

	return 0;
}

static inline int
xxx_kaapi_cuda_data_recv(
		kaapi_thread_context_t* thread,
		kaapi_metadata_info_t*  mdi,
		kaapi_data_t* h_dest,
		kaapi_data_t* d_src
		)
{
	const kaapi_address_space_id_t host_asid = 0UL;
	kaapi_cuda_mem_copy_dtoh( h_dest->ptr, &h_dest->view,
		d_src->ptr, &d_src->view );
//	_kaapi_metadata_info_set_valid( mdi, host_asid );
	return 0;
}

int kaapi_cuda_data_recv( 
	kaapi_thread_context_t* thread,
	kaapi_format_t*		   fmt,
	kaapi_task_t*              pc
)
{
	size_t count_params = kaapi_format_get_count_params(fmt, pc->sp );
	unsigned int i;
	kaapi_metadata_info_t*  mdi;
	const kaapi_address_space_id_t host_asid = 0UL;

#if KAAPI_VERBOSE
	fprintf(stdout, "[%s] CUDA params=%ld kid=%lu\n", __FUNCTION__,
		count_params,
		(unsigned long)kaapi_get_current_kid() );
	fflush( stdout );
#endif
	for ( i=0; i < count_params; i++ ) {
		kaapi_access_mode_t m = KAAPI_ACCESS_GET_MODE(
			kaapi_format_get_mode_param(fmt, i, pc->sp) );
		if (m == KAAPI_ACCESS_MODE_V) 
			continue;

		if( KAAPI_ACCESS_IS_WRITE(m) ) {
		kaapi_access_t access = kaapi_format_get_access_param( fmt,
			       	i, pc->sp );
		kaapi_data_t* handle = (kaapi_data_t*)access.data;
		mdi = handle->mdi;
		kaapi_data_t* dest = _kaapi_metadata_info_get_data( mdi,
			       	host_asid );
		xxx_kaapi_cuda_data_recv( thread, mdi, dest, handle);
		}
	}

	return 0;
}
