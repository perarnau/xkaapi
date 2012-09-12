
#include <stdio.h>

#include "kaapi_impl.h"
#include "kaapi_mem.h"
#include "kaapi_mem_data.h"
#include "kaapi_mem_host_map.h"

#if defined(KAAPI_USE_CUDA)

#if !defined(KAAPI_CUDA_NO_D2H)

static int
kaapi_memory_host_synchronize( void )
{
    KAAPI_EVENT_PUSH0( kaapi_get_current_processor(), kaapi_self_thread(), KAAPI_EVT_CUDA_CPU_SYNC_BEG );
    kaapi_cuda_proc_sync_all();
    KAAPI_EVENT_PUSH0( kaapi_get_current_processor(), kaapi_self_thread(), KAAPI_EVT_CUDA_CPU_SYNC_END );

#if defined(KAAPI_USE_CUDA)
   kaapi_assert_debug( kaapi_cuda_proc_all_isvalid( ) );
#endif

    return 0;
}

#endif /* !KAAPI_CUDA_NO_D2H */

#endif /* KAAPI_USE_CUDA */

int kaapi_memory_synchronize( void )
{
#if !defined(KAAPI_CUDA_NO_D2H) && defined(KAAPI_USE_CUDA)
    kaapi_memory_host_synchronize();
#endif
    return 0;
}

int kaapi_memory_synchronize_pointer( void *ptr )
{
    /* TODO */
    return 0;
}
