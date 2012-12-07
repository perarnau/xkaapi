/*
** xkaapi
** 
**
** Copyright 2012 INRIA.
**
** Contributors :
**
** thierry.gautier@inrialpes.fr
** Vincent.Danjean@ens-lyon.org
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
#include <stdio.h>
#include "kaapic.h"

#ifndef T
#  error T not defined
#endif
#define KAAPIC_TYPE(s) _KAAPIC_TYPE(s)
#define _KAAPIC_TYPE(s) KAAPIC_TYPE_##s

void body(T const n, T* result)
{
  *result = n;
}


int main()
{
  T result;
  kaapic_spawn_attr_t attr;
  
  kaapic_init(KAAPIC_START_ONLY_MAIN);

  kaapic_begin_parallel (KAAPIC_FLAG_DEFAULT);

  /* initialize default attribut */
  kaapic_spawn_attr_init(&attr);
  
  /* spawn the task */
  kaapic_spawn(&attr, 
      2,                /* number of arguments */
      (void(*)())body,  /* the entry point for the task */
      KAAPIC_MODE_V, KAAPIC_TYPE(KT), 1, (T)125,
      KAAPIC_MODE_W, KAAPIC_TYPE(KT), 1, &result
  );

  kaapic_sync();
  
  kaapic_end_parallel (KAAPIC_FLAG_DEFAULT);

  printf("The result is : %i/%f\n", (int)(uintptr_t)result, (double)(uintptr_t)result );
  kaapic_finalize();

  if (result == (T)125) {
    printf("Success\n");
  }
  
  return 0;
}
