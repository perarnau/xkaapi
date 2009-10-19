/*
** kaapi_attr_affinity.c
** xkaapi
** 
** Created on Tue Mar 31 15:20:37 2009
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

/* si thread user: cpuset = cpuset sur les processor kaapi, ie, bit i du cpuset == i-eme processor kaapi
   si thread kernel: cpuset = cpuset sur les cpu par defaut, ie, bit i du cpuset == i-eme processeur physique donne dans kaapi_param.h */
int kaapi_attr_setaffinity(kaapi_attr_t* attr, size_t cpusetsize, const cpu_set_t *cpuset)
{
  if (attr ==0) return EINVAL;
  attr->_cpuset = *cpuset;
  return 0;
}

int kaapi_attr_getaffinity(kaapi_attr_t* attr, size_t cpusetsize, cpu_set_t *cpuset)
{
  return ENOSYS;
}
