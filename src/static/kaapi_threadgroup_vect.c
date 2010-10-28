/*
** xkaapi
** 
** 
** Copyright 2010 INRIA.
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


/*
*/
int kaapi_vector_init( kaapi_vector_t* v, kaapi_vectentries_bloc_t* initbloc )
{
  v->firstbloc = initbloc;
  v->currentbloc = initbloc;
  v->allallocatedbloc = 0;
  if (initbloc !=0)
    v->currentbloc->pos = 0;
  return 0;
}



/*
*/
int kaapi_vector_destroy( kaapi_vector_t* v )
{
  while (v->allallocatedbloc !=0)
  {
    kaapi_vectentries_bloc_t* curr = v->allallocatedbloc;
    v->allallocatedbloc = curr->next;
    free (curr);
  }
  return 0;
}



/*
*/
kaapi_pidreader_t* kaapi_vector_pushback( kaapi_vector_t* v )
{ 
  kaapi_pidreader_t* entry; 
  /* allocate new entry */
  if (v->currentbloc == 0) 
  {
    v->currentbloc = malloc( sizeof(kaapi_vectentries_bloc_t) );
    v->currentbloc->next = v->allallocatedbloc;
    v->allallocatedbloc = v->currentbloc;
    v->currentbloc->pos = 0;
  }
  if (v->firstbloc ==0) v->firstbloc = v->currentbloc;
  
  entry = &v->currentbloc->data[v->currentbloc->pos];
  if (++v->currentbloc->pos == KAAPI_BLOCENTRIES_SIZE)
  {
    v->currentbloc = 0;
  }
  return entry;
}
