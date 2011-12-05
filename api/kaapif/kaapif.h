/*
 ** xkaapi
 ** 
 ** Created on Tue Mar 31 15:19:14 2009
 ** Copyright 2009 INRIA.
 **
 ** Contributors :
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


#ifndef KAAPIF_H_INCLUDED
# define KAAPIF_H_INCLUDED


#include <stdint.h>


#if defined(__cplusplus)
extern "C" {
#endif


/* kaapi fortran interface */
#define KAAPIF_SUCCESS 0
#define KAAPIF_ERR_FAILURE -1
#define KAAPIF_ERR_EINVAL -2
#define KAAPIF_ERR_UNIMPL -3

#if CONFIG_MAX_TID
extern int xxx_max_tid;
#endif
extern int xxx_seq_grain;
extern int xxx_par_grain;

extern int kaapif_init_(int32_t*);
extern int kaapif_finalize_(void);

extern double kaapif_get_time_(void);

extern int32_t kaapif_get_concurrency_(void);
extern int32_t kaapif_get_thread_num_(void);

extern void kaapif_set_max_tid_(int32_t*);
extern int32_t kaapif_get_max_tid_(void);

extern void kaapif_set_grains_(int32_t*, int32_t*);

extern int kaapif_foreach_(
  int32_t*,  /* first */
  int32_t*,  /* last */
  int32_t*,  /* NARGS */
  void (*)(int32_t*, int32_t*, int32_t*, ...), 
  ...  
);
extern int kaapif_foreach_with_format_(
  int32_t*,  /* first */
  int32_t*,  /* last */
  int32_t*,  /* NARGS */
  void (*)(int32_t*, int32_t*, int32_t*, ...), 
  ...
);

extern int kaapif_spawn_(
    int32_t*,    /* NARGS */
    void (*f)(), /* F */
    ...
);

extern void kaapif_sched_sync_(void);
extern int kaapif_begin_parallel_(void);
extern int kaapif_end_parallel_(int32_t*);


#if defined(__cplusplus)
}
#endif


#endif /* ! KAAPIF_H_INCLUDED */
