/*
** kaapi_condvar.h
** xkaapi
** 
**
** Copyright 2012 INRIA.
**
** Contributors :
**
** vincent.danjean@imag.fr
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

#include <pthread.h>
#include "kaapi_compiler.h"
#include "config.h"

#if (HAVE_FUTEX == 1)

typedef struct kproc_mutex kproc_mutex_t;
struct kproc_mutex {
	union {
		unsigned u;
		struct {
			unsigned char locked;
			unsigned char contended;
		} b;
	} st;
};

#define KPROC_MUTEX_INITIALIZER {0}

int __KA_INTERNAL kproc_mutex_init(kproc_mutex_t *m, const pthread_mutexattr_t *a);
int __KA_INTERNAL kproc_mutex_destroy(kproc_mutex_t *m);
int __KA_INTERNAL kproc_mutex_lock(kproc_mutex_t *m);
int __KA_INTERNAL kproc_mutex_unlock(kproc_mutex_t *m);
int __KA_INTERNAL kproc_mutex_trylock(kproc_mutex_t *m);

typedef struct kproc_cond kproc_cond_t;
struct kproc_cond
{
	kproc_mutex_t *m;
	int seq;
	int pad;
};

#define KPROC_COND_INITIALIZER {NULL, 0, 0}

int __KA_INTERNAL kproc_cond_init(kproc_cond_t *c, pthread_condattr_t *a);
int __KA_INTERNAL kproc_cond_destroy(kproc_cond_t *c);
int __KA_INTERNAL kproc_cond_wait(kproc_cond_t *c, kproc_mutex_t *m);
int __KA_INTERNAL kproc_cond_signal(kproc_cond_t *c);
int __KA_INTERNAL kproc_cond_broadcast(kproc_cond_t *c);

/* These condition variables does *not* reacquire the mutex
   after a wait operation */
typedef struct kproc_condunlock kproc_condunlock_t;
struct kproc_condunlock
{
	kproc_mutex_t *m;
	int seq;
	int pad;
};

#define KPROC_CONDUNLOCK_INITIALIZER {NULL, 0, 0}

int __KA_INTERNAL kproc_condunlock_init(kproc_condunlock_t *c, pthread_condattr_t *a);
int __KA_INTERNAL kproc_condunlock_destroy(kproc_condunlock_t *c);
int __KA_INTERNAL kproc_condunlock_wait(kproc_condunlock_t *c, kproc_mutex_t *m);
int __KA_INTERNAL kproc_condunlock_signal(kproc_condunlock_t *c);
int __KA_INTERNAL kproc_condunlock_broadcast(kproc_condunlock_t *c);

#else /* no futex */

#warning "Use Pthread mutex"
typedef struct {
	pthread_mutex_t pm;
} kproc_mutex_t;

#define KPROC_MUTEX_INITIALIZER {PTHREAD_MUTEX_INITIALIZER}

#define kproc_mutex_init(m,a) pthread_mutex_init(&((m)->pm),(a))
#define kproc_mutex_destroy(m) pthread_mutex_destroy(&((m)->pm))
#define kproc_mutex_lock(m) pthread_mutex_lock(&((m)->pm))
#define kproc_mutex_unlock(m) pthread_mutex_unlock(&((m)->pm))
#define kproc_mutex_trylock(m) pthread_mutex_trylock(&((m)->pm))

typedef struct {
	pthread_cond_t pc;
} kproc_cond_t;

#define KPROC_COND_INITIALIZER {PTHREAD_COND_INITIALIZER}

#define kproc_cond_init(c,a) pthread_cond_init(&((c)->pc),(a))
#define kproc_cond_destroy(c) pthread_cond_destroy(&((c)->pc))
#define kproc_cond_wait(c,m) pthread_cond_wait(&((c)->pc),&((m)->pm))
#define kproc_cond_signal(c) pthread_cond_signal(&((c)->pc))
#define kproc_cond_broadcast(c) pthread_cond_broadcast(&((c)->pc))

typedef struct {
	kproc_cond_t kc;
} kproc_condunlock_t;

#define KPROC_CONDUNLOCK_INITIALIZER {PTHREAD_COND_INITIALIZER}

#define kproc_condunlock_init(c,a) kproc_cond_init(&((c)->kc),(a))
#define kproc_condunlock_destroy(c) kproc_cond_destroy(&((c)->kc))
#define kproc_condunlock_wait(c,m) do { \
	kproc_cond_wait(&((c)->kc),(m)) ;	\
	kproc_mutex_unlock(m) ; \
      } while (0)
#define kproc_condunlock_signal(c) kproc_cond_signal(&((c)->kc))
#define kproc_condunlock_broadcast(c) kproc_cond_broadcast(&((c)->kc))

#endif
