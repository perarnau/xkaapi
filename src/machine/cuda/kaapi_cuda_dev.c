

#include <stdio.h>
#include <pthread.h>
#include <cuda_runtime_api.h>

#include "../../kaapi_impl.h"
#include "kaapi_cuda_proc.h"
#include "kaapi_cuda_dev.h"
#include "kaapi_cuda_ctx.h"

int
kaapi_cuda_dev_open( kaapi_cuda_proc_t* proc, unsigned int index )
{
    cudaError_t res;

    proc->index = index;

    cudaSetDevice( index );
    cudaDeviceReset();
    res= cudaSetDevice( index );
    if( res != cudaSuccess ) {
	fprintf( stdout, "%s: cudaSetDevice ERROR %d\n", __FUNCTION__, res );
	fflush( stdout );
	abort();
    }

    cudaFree(0);
    res = cudaGetDeviceProperties( &proc->deviceProp, index );
    if (res != cudaSuccess) {
	fprintf( stdout, "%s: cudaGetDeviceProperties ERROR %d\n", __FUNCTION__, res );
	fflush( stdout );
	abort();
    }

    /* 80% of total memory */
    proc->memory.total = 0.8*proc->deviceProp.totalGlobalMem;
    proc->memory.used= 0;
    proc->memory.beg = proc->memory.end = NULL;
    kaapi_big_hashmap_init( &proc->memory.kmem, 0 );  

    return 0;
}

void
kaapi_cuda_dev_close( kaapi_cuda_proc_t* proc )
{
    cudaDeviceReset();
}

kaapi_processor_t*
kaapi_cuda_mem_get_proc( void )
{
	unsigned int i;

	for (i=0; i<kaapi_count_kprocessors; ++i) {
		if( kaapi_all_kprocessors[i]->proc_type == KAAPI_PROC_TYPE_CUDA ) {
			return kaapi_all_kprocessors[i];
		}
	}

	return NULL;
}

kaapi_processor_t*
kaapi_cuda_get_proc_by_asid( kaapi_address_space_id_t asid )
{
	unsigned int i;

	for (i=0; i < kaapi_count_kprocessors; ++i) {
		if( (kaapi_all_kprocessors[i]->proc_type == KAAPI_PROC_TYPE_CUDA) && 
			 	(kaapi_all_kprocessors[i]->cuda_proc.asid == asid) ) {
			return kaapi_all_kprocessors[i];
		}
	}

	return NULL;
}
