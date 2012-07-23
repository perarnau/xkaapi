
#ifndef KAAPI_CUDA_TASKMOVEALLOC_H_INCLUDED
#define KAAPI_CUDA_TASKMOVEALLOC_H_INCLUDED

void kaapi_cuda_taskmove_body(void *sp, kaapi_thread_t * thread);

void kaapi_cuda_taskalloc_body(void *sp, kaapi_thread_t * thread);

#endif
