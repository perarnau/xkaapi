/*
 ** xkaapi
 ** 
 ** Created on Tue Feb 23 16:56:43 2010
 ** Copyright 2009 INRIA.
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

/**/
kaapi_data_version_t* kaapi_version_findasid_in( kaapi_version_t* ver, kaapi_address_space_id_t asid )
{
  if (ver ==0) return 0;
  kaapi_data_version_t* curr = ver->copies.front;
  while (curr !=0)
  {
    if (kaapi_memory_address_space_isequal(curr->asid, asid)) return curr;
    curr = curr->next;
  }
  if (kaapi_memory_address_space_isequal(ver->writer.asid, asid)) return &ver->writer;
  return 0;
}


/**/
kaapi_data_version_t* kaapi_version_findcopiesrmv_asid_in( kaapi_version_t* ver, kaapi_address_space_id_t asid )
{
  if (ver ==0) return 0;
  kaapi_data_version_t* curr = ver->copies.front;
  kaapi_data_version_t* prev = curr;
  while (curr !=0)
  {
    if (kaapi_memory_address_space_isequal(curr->asid, asid)) 
    {
      if (prev == curr) 
      {
        ver->copies.front = curr->next;
        if (ver->copies.front == 0) ver->copies.back = 0;
        prev = ver->copies.front;
        curr->next = 0;
        return curr;
      }

      prev->next = curr->next;
      if (ver->copies.back == curr) ver->copies.front = prev;
      return curr;
    }
    prev = curr;
    curr = curr->next;
  }
  return 0;
}



/**/
kaapi_data_version_t* kaapi_version_findtodelrmv_asid_in( kaapi_version_t* ver, kaapi_address_space_id_t asid )
{
  if (ver ==0) return 0;
  kaapi_data_version_t* curr = ver->todel.front;
  kaapi_data_version_t* prev = curr;
  while (curr !=0)
  {
    if (kaapi_memory_address_space_isequal(curr->asid, asid)) 
    {
      if (prev == curr) 
      {
        ver->todel.front = curr->next;
        if (ver->todel.front == 0) ver->todel.back = 0;
        prev = ver->todel.front;
        curr->next = 0;
        return curr;
      }

      prev->next = curr->next;
      if (ver->todel.back == curr) ver->todel.front = prev;
      return curr;
    }
    prev = curr;
    curr = curr->next;
  }
  return 0;
}


/**/
kaapi_comsend_t* kaapi_sendcomlist_find_tag( kaapi_taskbcast_arg_t* bcast, kaapi_comtag_t tag )
{
  kaapi_comsend_t* curr = &bcast->front;
  while (curr !=0)
  {
    if (curr->vertag == tag) return curr;
    curr = curr->next;
  }
  return 0;
}

kaapi_comsend_raddr_t* kaapi_sendcomlist_find_asid( kaapi_comsend_t* com, kaapi_address_space_id_t asid )
{
  kaapi_comsend_raddr_t* curr = &com->front;
  while (curr !=0)
  {
    if (kaapi_memory_address_space_isequal(curr->asid, asid)) 
      return curr;
    curr = curr->next;
  }
  return 0;
}


/**/
kaapi_comrecv_t* kaapi_recvcomlist_find_tag( kaapi_comlink_t* recvl, kaapi_comtag_t tag )
{
  while (recvl !=0)
  {
    if (recvl->u.recv->tag == tag) return recvl->u.recv;
    recvl = recvl->next;
  }
  return 0;
}

/**
*/
kaapi_comsend_raddr_t* kaapi_threadgroup_findsend_tagtid( kaapi_comaddrlink_t* list, kaapi_comtag_t tag, int tid )
{
  while (list !=0)
  {
    if ((tag == list->tag) && (kaapi_memory_address_space_getuser(list->send->asid) == tid)) 
      return list->send;
    list = list->next;
  }
  return 0;
}

