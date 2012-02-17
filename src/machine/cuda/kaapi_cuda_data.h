
#ifndef KAAPI_CUDA_DATA_H_INCLUDED
#define KAAPI_CUDA_DATA_H_INCLUDED

int kaapi_cuda_data_allocate( 
	kaapi_format_t*		   fmt,
	void*			sp
);

int kaapi_cuda_data_send( 
	kaapi_format_t*		   fmt,
	void*			sp
);

#if	KAAPI_CUDA_MEM_ALLOC_MANAGER
int 
kaapi_cuda_data_check( void );
#endif

/* ** Memory system **
    This method is called by a CUDA thread to synchronize the kdata parameter.
It checks if the data is valid on the current kproc, otherwise searches for a
valid copy on the asids of the system.
*/
int
kaapi_cuda_data_sync_device( kaapi_data_t* kdata );

/* ** Memory system **
   This method is called from a host thread to synchronize the kdata parameter.
It checks if the data is valid on the current kproc, otherwise search for a
valid copy on the GPUs.
*/
int
kaapi_cuda_data_sync_host( kaapi_data_t* kdata );

#endif
