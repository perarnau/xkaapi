/*
** kaapi_mt_topology.c
** xkaapi
** 
** Created on Tue Mar 31 15:19:03 2009
** Copyright 2009 INRIA.
**
** Contributors :
**
** christophe.laferriere@imag.fr
** thierry.gautier@imag.fr
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
#include <stdlib.h>

/*
*/
kaapi_uint32_t kaapi_hierarchy_level = 0;

/*
*/
kaapi_neighbors_t** kaapi_neighbors = 0;


#if defined(IDKOIFF)
/* here is assumed that the hypertransport interconnect of idkoiff
   if the following (here is the core identifier):
   --O,1---4,5---8,9---12,13
      |     |      \    /|
      |     |       \  / |
      |     |        \/  |
      |     |        /\  |
      |     |       /  \ |
      |     |      /    \|
   --2,3---6,7---10,11---14,15
   TODO: correct this diagram
*/
#endif

/*
*/
#define MYMAC /* a core2 duo */

/** TODO: this method should be implemented in order to specify a topology.
    May be it would be better to specify it into a file given at runtime (it will allow 
    to make more test without recompiling the library).
    If we consider the above topology for idkoiff (should be verify), we may have been
    interested in viewing the following kind of hierarchy:
    1/ 2 levels hierarchy: 
          - the cores on the same socket accessing to a shared cache (?) and a fast memory bank
          - all the cores that share the main memory
    2/ 4 levels hierarchy:
          - cores on the same socket
          - cores at one hop following the topology
          - cores at two hops
          - cores at three hops
    The format for a file may be something like this:
    <list of cores>: <list of cores at level 0>, <list of cores at level 1>, <list of cores at level 2>, etc...
    where list of cores:= '{' integer [',' integer]* '}.

    For instance kind 1/:
        {0, 1}: {0, 1}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15}
        {2, 3}: {2, 3}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15}
        ...
    For instance kind 2/:
        {0, 1}: {0, 1}, {0, 1, 2, 3, 4, 5}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15}
        ...
        {4, 5}: {4, 5}, {0, 1, 4, 5, 6, 7, 8, 9}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15}
        ...
        
    This topology information is only based on logical identification of cores (identifiers range from 0 to N-1, where
    N is the maximal number of ressources). If this logical identification does not map the physical / system dependent
    numbering, a additional mapping of logical identifiers from 0-N to the physical identifiers should be provided:
        <logical number> -> <physical identifier>
    [TG: I hope that physical identifier is always an integer....]
*/
int kaapi_setup_topology(void)
{
  int i;
  
  /* this function should be called during the initialization of kaapi, after havind decoding the
     default parameters and the array of kaapi_all_kprocessors has tobe already defined at correct
     memory location.
  */
}
