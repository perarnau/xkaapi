
#include <stdio.h>

#include "kaapi_impl.h"

#if defined(KAAPI_USE_CUDA)
#include <cuda_runtime_api.h>
#endif

int kaapi_memory_register( void* ptr, const size_t size )
{
#if defined(KAAPI_USE_CUDA)
    const cudaError_t res= cudaHostRegister( ptr, size, cudaHostRegisterPortable );
    if (res != cudaSuccess) {
	    fprintf( stdout, "[%s] ERROR (%d) ptr=%p size=%lu kid=%lu\n",
			    __FUNCTION__, res,
			    ptr, size,
			    (long unsigned int)kaapi_get_current_kid() ); 
	    fflush( stdout );
    }

    return res;
#else
    return 0;
#endif
}

void kaapi_memory_unregister( void* ptr )
{
    cudaHostUnregister( ptr );
}
