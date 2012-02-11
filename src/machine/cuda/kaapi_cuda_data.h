
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

int kaapi_cuda_data_recv( 
	kaapi_format_t*		   fmt,
	void*	              sp
);

int kaapi_cuda_data_send_ptr( 
	kaapi_format_t*		   fmt,
	void*			sp
);

int kaapi_cuda_data_recv_ptr( 
	kaapi_format_t*		   fmt,
	void*	              sp
);

#if	KAAPI_CUDA_MEM_ALLOC_MANAGER
int 
kaapi_cuda_data_check( void );
#endif

#endif
