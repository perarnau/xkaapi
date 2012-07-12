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
#include <string.h>
#include <stdbool.h>
#include <ctype.h>
#include <errno.h>

#include "kaapi_impl.h"
#include "kaapic_impl.h"

unsigned long komp_env_nthreads = 0;
unsigned long omp_max_active_levels = 0;

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

#define __append_string_with_format(kaapi_cpuset_string, format, ...)  \
do {                                                                   \
  char tmp[16];                                                        \
  sprintf (tmp, format, __VA_ARGS__);                                  \
  strcat (kaapi_cpuset_string, tmp);                                   \
} while (0)


/* This function parses the content of the GOMP_CPU_AFFINITY 
   environment variable to generate something that fits the semantic 
   of the KAAPI_CPUSET environment variable. */
static void
komp_parse_cpu_affinity (void)
{
  char *cpu_affinity = getenv ("GOMP_CPU_AFFINITY");
  if (cpu_affinity == NULL)
    return;
  
  char kaapi_cpuset[512];
  bzero (kaapi_cpuset, 512 * sizeof (char));
  
  char *cpuset_string = NULL;  
  while ((cpuset_string = strsep (&cpu_affinity, " \t")) != NULL)
  {
    char *first_cpu = strsep (&cpuset_string, "-");
    char *last_cpu = strsep (&cpuset_string, ":");
    char *stride = cpuset_string;

    if (stride != NULL)
    {
      int begin = atoi (first_cpu);
      int end = atoi (last_cpu);
      int hop = atoi (stride);
      
      for (int i = begin; i < end; i += hop)
        __append_string_with_format (kaapi_cpuset, "%d,", i); 

    } else {
      if (last_cpu != NULL)
        __append_string_with_format (kaapi_cpuset, "%s:%s,", first_cpu, last_cpu); 
      else 
        __append_string_with_format (kaapi_cpuset, "%s,", first_cpu); 
    }
  }
  kaapi_cpuset[strlen (kaapi_cpuset) - 1] = '\0';
  setenv ("KAAPI_CPUSET", kaapi_cpuset, 1);
}

static void __attribute__ ((constructor))  
initialize_lib (void) 
{
  if (parse_unsigned_long ("OMP_NUM_THREADS", &komp_env_nthreads))
  {
#if 0 /* This may break programs that use more threads than cores. */ 
    /* Kaapi inherits OMP_NUM_THREADS */
    setenv ("KAAPI_CPUCOUNT", getenv ("OMP_NUM_THREADS"), 1);
#endif
  }
  
  /* Turn GOMP_CPU_AFFINITY into KAAPI_CPUSET. */
  komp_parse_cpu_affinity ();

  if (!parse_unsigned_long ("OMP_MAX_ACTIVE_LEVELS", &omp_max_active_levels))
    omp_max_active_levels = KAAPI_MAX_RECCALL;

  kaapic_init (KAAPIC_START_ONLY_MAIN);
}

static void __attribute__ ((destructor))
finalize_lib (void)
{
  kaapic_finalize ();
}
