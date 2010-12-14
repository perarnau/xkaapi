/*
** kaapi_mt_register_procs.h
** 
** Created on Jun 23 2010
** Copyright 2009 INRIA.
**
** Contributors :
**
** thierry.gautier@inrialpes.fr
** fabien.lementec@imag.fr
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


#include <stdlib.h>
#include "kaapi_impl.h"
#include "../common/kaapi_procinfo.h"

static int make_identity_procinfo_list(kaapi_procinfo_list_t* kpl, unsigned int count)
{
  /* make an identity cpu list for the range [0, count[ */
  unsigned int cpu_index;

  for (cpu_index = 0; cpu_index < count; ++cpu_index)
  {
    kaapi_procinfo_t* const kpi = kaapi_procinfo_alloc();
    if (kpi == NULL)
      return -1;

    kpi->bound_cpu = cpu_index;
    kpi->proc_type = KAAPI_PROC_TYPE_CPU;
    kpi->proc_index = cpu_index;
    kpi->bound_cpu = cpu_index;

    kaapi_procinfo_list_add(kpl, kpi);
  }
  
  return 0;
}


/* exported */
int kaapi_mt_register_procs(kaapi_procinfo_list_t* kpl)
{
  const char* const cpuset_str = getenv("KAAPI_CPUSET");
  const char* const cpucount_str = getenv("KAAPI_CPUCOUNT");

  /* KAAPI_CPUCOUNT */
  if (cpucount_str != NULL)
  {
    kaapi_default_param.cpucount = atoi(getenv("KAAPI_CPUCOUNT"));
    if (kaapi_default_param.cpucount > KAAPI_MAX_PROCESSOR_LIMIT)
      kaapi_default_param.cpucount = kaapi_default_param.syscpucount;
  }

  /* KAAPI_CPUSET */
  if (cpuset_str != NULL)
  {
    const int err = kaapi_procinfo_list_parse_string(
            kpl, cpuset_str, KAAPI_PROC_TYPE_CPU, 
            kaapi_default_param.cpucount
    );
    if (err == -1) return -1;
    kaapi_default_param.use_affinity = 1;
  }
  else /* identity set */
  {
    if (make_identity_procinfo_list(kpl, kaapi_default_param.cpucount))
      return -1;
  }

  return 0;
}
