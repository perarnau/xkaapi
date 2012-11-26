
#include <stdio.h>

#include "kaapi_impl.h"

#if defined(KAAPI_USE_CUDA)
#include "machine/cuda/kaapi_cuda_mem.h"
#endif

int kaapi_memory_register( void* ptr, kaapi_memory_view_t view )
{
#if defined(KAAPI_USE_CUDA)
    kaapi_cuda_mem_register_( ptr,
		    kaapi_memory_view_size(&view) );
#endif
    return 0;
}

void kaapi_memory_unregister( void* ptr )
{
#if defined(KAAPI_USE_CUDA)
    kaapi_cuda_mem_unregister_( ptr );
#endif
}
