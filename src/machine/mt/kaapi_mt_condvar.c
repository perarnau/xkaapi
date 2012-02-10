/*
** kaapi_condvar.c
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

#include <time.h>
#ifndef _GNU_SOURCE
#  define _GNU_SOURCE
#endif

#if (HAVE_FUTEX==1)

#include <unistd.h>
#include <sys/syscall.h>
#include <linux/futex.h>
#include <errno.h>
#include <pthread.h>
#include <assert.h>
#include <limits.h>

#include "machine/mt/kaapi_mt_condvar.h"


#define atomic_xadd(P, V) __sync_fetch_and_add((P), (V))
#define cmpxchg(P, O, N) __sync_val_compare_and_swap((P), (O), (N))
#define atomic_inc(P) __sync_add_and_fetch((P), 1)
#define atomic_dec(P) __sync_add_and_fetch((P), -1) 
#define atomic_add(P, V) __sync_add_and_fetch((P), (V))
#define atomic_set_bit(P, V) __sync_or_and_fetch((P), 1<<(V))
#define atomic_clear_bit(P, V) __sync_and_and_fetch((P), ~(1<<(V)))

#if defined(__i386__) || defined(__x86_64__)

/* Atomic exchange of 32 bits */
static inline unsigned xchg_32(void *ptr, unsigned x)
{
	__asm__ __volatile__("xchgl %0,%1"
				:"=r" ((unsigned) x)
				:"m" (*(volatile unsigned *)ptr), "0" (x)
				:"memory");

	return x;
}
#else
#  warning Add required assembly instructions for this architecture
#  error or disable futex use
#endif

static inline int sys_futex(
  void *addr1, 
  int op, 
  int val1, 
  struct timespec *timeout, 
  void *addr2, 
  int val3
)
{
	return syscall(SYS_futex, addr1, op, val1, timeout, addr2, val3);
}

int kproc_mutex_init(kproc_mutex_t *m, const pthread_mutexattr_t *a)
{
	assert(a==NULL);
	m->st.u = 0;
	return 0;
}

int kproc_mutex_destroy(kproc_mutex_t *m)
{
	/* Do nothing */
	(void) m;
	return 0;
}

int kproc_mutex_lock(kproc_mutex_t *m)
{
	int i;
	
	/* Try to grab lock */
	for (i = 0; i < 100; i++)
	{
		unsigned u=m->st.u;
		if ( (u&1) == 0 &&
		     cmpxchg(&m->st.u, u, u|1)==u) return 0;

		kaapi_slowdown_cpu();
	}

	/* Have to sleep */
	while (xchg_32(&m->st.u, 257) & 1)
	{
		sys_futex(&m->st.u, FUTEX_WAIT_PRIVATE, 257, NULL, NULL, 0);
	}
	
	return 0;
}

int kproc_mutex_unlock(kproc_mutex_t *m)
{
	int i;
	
	/* Locked and not contended */
	if ((m->st.u == 1) && (cmpxchg(&m->st.u, 1, 0) == 1)) return 0;
	
	/* Unlock */
	m->st.b.locked = 0;
	
	kaapi_mem_barrier();
	
	/* Spin and hope someone takes the lock */
	for (i = 0; i < 200; i++)
	{
		if (m->st.b.locked) return 0;
		
		kaapi_slowdown_cpu();
	}
	
	/* We need to wake someone up */
	m->st.b.contended = 0;
	
	sys_futex(&m->st.u, FUTEX_WAKE_PRIVATE, 1, NULL, NULL, 0);
	
	return 0;
}

int kproc_mutex_trylock(kproc_mutex_t *m)
{
	unsigned u = m->st.u;
	if ( (u&1) == 0 &&
	     cmpxchg(&m->st.u, u, u|1)==u) return 0;
	return EBUSY;
}

/*****************************************************************/
int kproc_cond_init(kproc_cond_t *c, pthread_condattr_t *a)
{
	assert(a==NULL);
	
	c->m = NULL;
	
	/* Sequence variable doesn't actually matter, but keep valgrind happy */
	c->seq = 0;
	
	return 0;
}

int kproc_cond_destroy(kproc_cond_t *c)
{
	/* No need to do anything */
	(void) c;
	return 0;
}

int kproc_cond_signal(kproc_cond_t *c)
{
	/* We are waking someone up */
	atomic_add(&c->seq, 1);
	
	/* Wake up a thread */
	sys_futex(&c->seq, FUTEX_WAKE_PRIVATE, 1, NULL, NULL, 0);
	
	return 0;
}

int kproc_cond_broadcast(kproc_cond_t *c)
{
	kproc_mutex_t *m = c->m;
	
	/* No mutex means that there are no waiters */
	if (!m) return 0;
	
	/* We are waking everyone up */
	atomic_add(&c->seq, 1);
	
	/* Wake one thread, and requeue the rest on the mutex */
	sys_futex(&c->seq, FUTEX_REQUEUE_PRIVATE, 1, (void *) INT_MAX, m, 0);
	
	return 0;
}

int kproc_cond_wait(kproc_cond_t *c, kproc_mutex_t *m)
{
	int seq = c->seq;

	if (c->m != m)
	{
		/* Atomically set mutex inside kproc_cond_t */
		void* res __attribute__((unused))
			= cmpxchg(&c->m, NULL, m);
		if (c->m != m) return EINVAL;
	}
	
	kproc_mutex_unlock(m);
	
	sys_futex(&c->seq, FUTEX_WAIT_PRIVATE, seq, NULL, NULL, 0);
	
	while (xchg_32(&m->st.u, 257) & 1)
	{
		sys_futex(&m->st, FUTEX_WAIT_PRIVATE, 257, NULL, NULL, 0);
	}
		
	return 0;
}

/*****************************************************************/
int kproc_condunlock_init(kproc_condunlock_t *c, pthread_condattr_t *a)
{
	assert(a==NULL);
	
	c->m = NULL;
	
	/* Sequence variable doesn't actually matter, but keep valgrind happy */
	c->seq = 0;
	
	return 0;
}

int kproc_condunlock_destroy(kproc_condunlock_t *c)
{
	/* No need to do anything */
	(void) c;
	return 0;
}

int kproc_condunlock_signal(kproc_condunlock_t *c)
{
	/* We are waking someone up */
	atomic_add(&c->seq, 1);
	
	/* Wake up a thread */
	sys_futex(&c->seq, FUTEX_WAKE_PRIVATE, 1, NULL, NULL, 0);
	
	return 0;
}

int kproc_condunlock_broadcast(kproc_condunlock_t *c)
{
	kproc_mutex_t *m = c->m;
	
	/* No mutex means that there are no waiters */
	if (!m) return 0;
	
	/* We are waking everyone up */
	atomic_add(&c->seq, 1);
	
	/* Wake all threads, they wont block on mutex */
	sys_futex(&c->seq, FUTEX_WAKE_PRIVATE, INT_MAX, NULL, NULL, 0);
	
	return 0;
}

int kproc_condunlock_wait(kproc_condunlock_t *c, kproc_mutex_t *m)
{
	int seq = c->seq;

	if (c->m != m)
	{
		/* Atomically set mutex inside kproc_condunlock_t */
		void* res __attribute__((unused))
			= cmpxchg(&c->m, NULL, m);
		if (c->m != m) return EINVAL;
	}	
	kproc_mutex_unlock(m);
	
	sys_futex(&c->seq, FUTEX_WAIT_PRIVATE, seq, NULL, NULL, 0);
	
	/* No lock reaquirement */

	return 0;
}

#endif
