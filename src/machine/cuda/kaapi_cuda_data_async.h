/*
** kaapi_cuda_data_async.h
** xkaapi
** 
** Created on Jan 2012
** Copyright 2010 INRIA.
**
** Contributors :
**
** thierry.gautier@inrialpes.fr
** Joao.Lima@imag.fr
** 
** This software is a computer program whose purpose is to execute
** multithreaded computation with data flow synchronization between
** threads.
** 
** This software is governed by the CeCILL-C license under French law
** and abiding by the rules of distribution of free software.  You can
** use, modify and/ or redistribute the software under the terms of
** the CeCILL-C license as circulated by CEA, CNRS and INRIA at the
** following URL "http://www.cecill.info".
** 
** As a counterpart to the access to the source code and rights to
** copy, modify and redistribute granted by the license, users are
** provided only with a limited warranty and the software's author,
** the holder of the economic rights, and the successive licensors
** have only limited liability.
** 
** In this respect, the user's attention is drawn to the risks
** associated with loading, using, modifying and/or developing or
** reproducing the software by the user in light of its specific
** status of free software, that may mean that it is complicated to
** manipulate, and that also therefore means that it is reserved for
** developers and experienced professionals having in-depth computer
** knowledge. Users are therefore encouraged to load and test the
** software's suitability as regards their requirements in conditions
** enabling the security of their systems and/or data to be ensured
** and, more generally, to use and operate it in the same conditions
** as regards security.
** 
** The fact that you are presently reading this means that you have
** had knowledge of the CeCILL-C license and that you accept its
** terms.
** 
*/

#ifndef KAAPI_CUDA_DATA_ASYNC_H_INCLUDED
#define KAAPI_CUDA_DATA_ASYNC_H_INCLUDED

int kaapi_cuda_data_async_allocate( 
	kaapi_cuda_stream_t* kstream,
	kaapi_tasklist_t*   tasklist,
	kaapi_taskdescr_t*  td
);

int kaapi_cuda_data_async_send( 
	kaapi_cuda_stream_t* kstream,
	kaapi_tasklist_t*   tasklist,
	kaapi_taskdescr_t*  td
);

int kaapi_cuda_data_async_recv( 
	kaapi_cuda_stream_t* kstream,
	kaapi_tasklist_t*   tasklist,
	kaapi_taskdescr_t*  td
);

#if	KAAPI_CUDA_MEM_ALLOC_MANAGER
int 
kaapi_cuda_data_async_check( void );
#endif

/* ** Memory system **
    This method is called by a CUDA thread to synchronize the kdata parameter.
It checks if the data is valid on the current kproc, otherwise searches for a
valid copy on the asids of the system.
*/
int
kaapi_cuda_data_async_sync_device( kaapi_data_t* kdata );

/* ** Memory system **
   This method is called from a host thread to synchronize the kdata parameter.
It checks if the data is valid on the current kproc, otherwise search for a
valid copy on the GPUs.
*/
int
kaapi_cuda_data_async_sync_host( kaapi_data_t* kdata, cudaStream_t stream );

#endif /* KAAPI_CUDA_DATA_ASYNC_H_INCLUDED */
