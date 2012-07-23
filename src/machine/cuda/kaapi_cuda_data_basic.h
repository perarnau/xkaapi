
#ifndef KAAPI_CUDA_DATA_BASIC_H_INCLUDED
#define KAAPI_CUDA_DATA_BASIC_H_INCLUDED

int kaapi_cuda_data_basic_allocate(kaapi_format_t * fmt, void *sp);

int kaapi_cuda_data_basic_send(kaapi_format_t * fmt, void *sp);

int kaapi_cuda_data_basic_recv(kaapi_format_t * fmt, void *sp);

#endif				/* KAAPI_CUDA_DATA_BASIC_H_INCLUDED */
