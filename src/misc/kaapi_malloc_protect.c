/*
** kaapi_abort.c
** xkaapi
** 
**
** Copyright 2009,2010,2011,2012 INRIA.
**
** Contributors :
**
** thierry.gautier@inrialpes.fr
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
#include "kaapi_impl.h"

#include <sys/mman.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdio.h>

#define NUMBER_GUARD_PAGES 1
#define MAGIC_NUMBER 12345678901234567890UL

void* kaapi_alloc_protect( size_t size )
{
    char* tmp;
    int err;
    size_t i;
    size_t psz = getpagesize();
    size_t sz = 2*NUMBER_GUARD_PAGES*psz + ((size + psz-1)/psz)*psz;
    tmp = (char*)mmap( 0, sz, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANON, -1, (off_t)0 );
    if (tmp == (char*)-1)
    {
        fprintf(stderr, "[kaapi_alloc_protect] cannot allocate memory\n");
        return 0;
    }

    /* write magic number on the first page */
    size_t* data = (size_t*)tmp;
    data[0] = sz;
    data[1] = size;
    for (i=2; i< NUMBER_GUARD_PAGES*psz / sizeof(size_t); ++i)
        data[i] = (size_t)MAGIC_NUMBER;

    /* protect first page */
    err = mprotect( tmp, NUMBER_GUARD_PAGES*psz, PROT_NONE );
    if (err !=0) {
        fprintf(stderr, "[kaapi_alloc_protect] cannot protect reserved page\n");
        goto return_free;
    }

    /* protect last page */
    err = mprotect( tmp+sz-NUMBER_GUARD_PAGES*psz, NUMBER_GUARD_PAGES*psz, PROT_NONE );
    if (err !=0) {
        fprintf(stderr, "[kaapi_alloc_protect] cannot protect reserved page\n");
        goto return_free;
    }
    /* user data begins at tmp+psz */
    return tmp+NUMBER_GUARD_PAGES*psz;

return_free:
    munmap( tmp, sz );
    return 0;
}


void kaapi_free_protect( void* p )
{
    if (p ==0) return;
    size_t psz = getpagesize();
    char* tmp = ((char*)p) - NUMBER_GUARD_PAGES*psz;
    int err;
    size_t i;
    /* suppress protection */
    err = mprotect( tmp, NUMBER_GUARD_PAGES*psz, PROT_READ|PROT_WRITE);
    if (err !=0) {
        fprintf(stderr, "[kaapi_alloc_protect] cannot suppress protection on reserved page\n");
        abort();
        return;
    }
    size_t* data = (size_t*)tmp;

    for (i=2; i< NUMBER_GUARD_PAGES*psz / sizeof(size_t); ++i)
        if (data[i] != (size_t)MAGIC_NUMBER)
        {
            fprintf(stderr,"[kaapi_alloc_protect] data seems corrumpted\n");
            abort();
            break;
        }

    /* */

    err = mprotect( tmp+data[0]-NUMBER_GUARD_PAGES*psz, NUMBER_GUARD_PAGES*psz, PROT_READ|PROT_WRITE);
    if (err !=0) {
        fprintf(stderr, "[kaapi_alloc_protect] cannot suppress protection on reserved page\n");
        abort();
        return;
    }
    /* free */
    munmap( tmp, data[0] );
}


void* kaapi_realloc_protect(void *ptr, size_t size)
{
  int err;
  size_t psz = getpagesize();
  char* tmp = ((char*)ptr) - NUMBER_GUARD_PAGES*psz;
  void* retval = kaapi_alloc_protect( size );

  if (ptr ==0) 
    return retval;

  /* suppress protection */
  err = mprotect( tmp, NUMBER_GUARD_PAGES*psz, PROT_READ|PROT_WRITE);
  if (err !=0) {
    fprintf(stderr, " [kaapi_alloc_protect] cannot suppress protection on reserved page\n");
    abort();
    return 0;
  }
  size_t* data = (size_t*)tmp;
  memcpy(retval, ptr, data[1]);
  err = mprotect( tmp, NUMBER_GUARD_PAGES*psz, PROT_NONE);
  if (err !=0) {
    fprintf(stderr, "[kaapi_alloc_protect] cannot suppress protection on reserved page\n");
    abort();
    return 0;
  }
  kaapi_free_protect( ptr );

  return retval;
}

void* kaapi_memalign_protect(size_t alignment, size_t size)
{
  return kaapi_alloc_protect( size );
}
