/*
** xkaapi
** 
**
** Copyright 2009 INRIA.
**
** Contributors :
**
** thierry.gautier@inrialpes.fr
** francois.broquedis@imag.fr
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
#include <stdio.h>
#include <stdbool.h>
#include <ctype.h>
#include <errno.h>
#include "kaapic.h"

int gomp_nthreads_var = 0;

/* Parse an unsigned long environment varible.  Return true if one was
   present and it was successfully parsed.  */

static bool
parse_unsigned_long (const char *name, unsigned long *pvalue)
{
  char *env, *end;
  unsigned long value;

  env = getenv (name);
  if (env == NULL)
    return false;

  while (isspace ((unsigned char) *env))
    ++env;
  if (*env == '\0')
    goto invalid;

  errno = 0;
  value = strtoul (env, &end, 10);
  if (errno || (long) value <= 0)
    goto invalid;

  while (isspace ((unsigned char) *end))
    ++end;
  if (*end != '\0')
    goto invalid;

  *pvalue = value;
  return true;

 invalid:
  fprintf (stderr, "Invalid value for environment variable %s", name);
  return false;
}


static void __attribute__ ((constructor))  
initialize_lib (void) 
{
  unsigned long env_nthreads = 0;
  
  if (parse_unsigned_long ("OMP_NUM_THREADS", &env_nthreads))
  {
    /* Kaapi inherits OMP_NUM_THREADS */
    setenv ("KAAPI_CPUCOUNT", getenv ("OMP_NUM_THREADS"), 1);
  }
  /* here to do: convert GOM_AFFINITY to KAAPI_CPUSET */

  kaapic_init (KAAPIC_START_ONLY_MAIN);

  gomp_nthreads_var = (env_nthreads > 0) ? env_nthreads : kaapic_get_concurrency ();  

  kaapic_begin_parallel ();
}

static void __attribute__ ((destructor))
finalize_lib (void)
{
  kaapic_end_parallel (0);
  kaapic_finalize ();
}
