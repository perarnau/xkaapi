
#include <stdio.h>

#include "cublas_v2.h"

#include "kaapi_impl.h"
#include "kaapi_cuda_cublas.h"
#include "kaapi_cuda_proc.h"

int kaapi_cuda_cublas_init( kaapi_cuda_proc_t *proc )
{
	cublasStatus_t status = cublasCreate( &proc->ctx.handle );
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf( stderr, "[%s] CUBLAS ERROR: %u\n", __FUNCTION__, status);
		return -1;
	}
	//cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_DEVICE);
	cublasSetPointerMode( proc->ctx.handle, CUBLAS_POINTER_MODE_HOST);
	
#if KAAPI_CUDA_ASYNC
	status= cublasSetStream( proc->ctx.handle, kaapi_cuda_kernel_stream() );
	if (status != CUBLAS_STATUS_SUCCESS) {
		fprintf( stderr, "[%s] CUBLAS ERROR: %u\n", __FUNCTION__, status);
		return -1;
	}
#endif

	return 0;
}

void kaapi_cuda_cublas_finalize( kaapi_cuda_proc_t *proc )
{
	cublasDestroy( proc->ctx.handle );
}

cublasHandle_t kaapi_cuda_cublas_handle( void )
{
	return (kaapi_get_current_processor()->cuda_proc.ctx.handle);
}
