/*
** kaapi_init.c
** xkaapi
** 
** Created on Tue Mar 31 15:19:03 2009
** Copyright 2009 INRIA.
**
** Contributors :
**
** christophe.laferriere@imag.fr
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
#include <stdlib.h>
#include <sys/types.h>
#include <sys/sysctl.h>
#include <unistd.h>


/*
*/
kaapi_rtparam_t default_param;

/**
*/
extern int kaapi_setup_param( int argc, char** argv )
{
  /* compute the number of cpu of the system */
#if defined(KAAPI_USE_LINUX)
  default_param.syscpucount = sysconf(_SC_NPROCESSORS_CONF);
#elif defined(KAAPI_USE_APPLE)
  {
    int mib[2];
    size_t len;
    mib[0] = CTL_HW;
    mib[1] = HW_NCPU;
    len = sizeof(default_param.syscpucount);
    sysctl(mib, 2, &default_param.syscpucount, &len, 0, 0);
  }
#else
  #warning "Could not compute number of physical cpu of the system. Default value==1"
  default_param.syscpucount = 1;
#endif
  /* adjust system limit, if library is compiled with greather number of processors that available */
  if (default_param.syscpucount < KAAPI_MAX_PROCESSOR)
    default_param.syscpucount = KAAPI_MAX_PROCESSOR;
    
  /* Set default values */
  default_param.cpucount  = default_param.syscpucount;
  default_param.stacksize = 8*4096;
  
  /* Get values from environment variable */
  if (getenv("KAAPI_STACKSIZE") !=0)
  {
    default_param.stacksize = atoi(getenv("KAAPI_STACKSIZE"));
  }
  if (getenv("KAAPI_CPUCOUNT") !=0)
  {
    default_param.cpucount = atoi(getenv("KAAPI_CPUCOUNT"));
  }
  
  /* default workstealing selection function */
  default_param.wsselect = &kaapi_sched_select_victim_rand;

  /* TODO: here parse command line option */
  
  return 0;
}
