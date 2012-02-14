
#include <stdio.h>

#include "kaapi_impl.h"

#if defined(KAAPI_USE_CUDA)
#include "machine/cuda/kaapi_cuda_mem.h"
#endif

int kaapi_memory_register( void* ptr, const size_t size )
{
#if defined(KAAPI_USE_CUDA)
    return kaapi_cuda_mem_register_( ptr, size );
#else
    return 0;
#endif
}

void kaapi_memory_unregister( void* ptr )
{
#if defined(KAAPI_USE_CUDA)
    kaapi_cuda_mem_unregister_( ptr );
#endif
}
