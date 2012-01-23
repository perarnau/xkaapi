
#ifndef KAAPI_CUDA_DATA_H_INCLUDED
#define KAAPI_CUDA_DATA_H_INCLUDED

int kaapi_cuda_data_send( 
	kaapi_thread_context_t* thread,
//	kaapi_taskdescr_t*         td,
	kaapi_format_t*		   fmt,
	kaapi_task_t*              pc
);

int kaapi_cuda_data_recv( 
	kaapi_thread_context_t* thread,
//	kaapi_taskdescr_t*         td,
	kaapi_format_t*		   fmt,
	kaapi_task_t*              pc
);

#endif
