
#include "kaapi_impl.h"
#include "kaapi_mem.h"

extern kaapi_big_hashmap_t kmem_hm;

void
kaapi_mem_init( void )
{
#if KAAPI_VERBOSE
    fprintf( stdout, "[%s] \n", __FUNCTION__ );
    fflush(stdout);
#endif
    kaapi_big_hashmap_init( &kmem_hm, 0 );  
}

