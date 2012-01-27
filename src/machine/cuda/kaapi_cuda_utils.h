
#if 0

#include "kaapi_cuda_proc.h"

void kaapi_cuda_ctx_push( kaapi_cuda_proc_t* proc );

void kaapi_cuda_ctx_pop( kaapi_cuda_proc_t* proc );

int kaapi_cuda_dev_open( kaapi_cuda_proc_t* proc, unsigned int index );

void kaapi_cuda_dev_close( kaapi_cuda_proc_t* proc );

kaapi_cuda_proc_t* kaapi_cuda_mem_get_proc( void );

kaapi_cuda_proc_t* kaapi_cuda_get_proc_by_asid( kaapi_address_space_id_t asid );

#endif /* ! KAAPI_CUDA_UTILS_H_INCLUDED */
