#if (defined(__sparc_v9__) && (defined(__fcc_version) || defined(__FCC_VERSION)))

#ifndef KAAPI_ARCH_KCOMPUTERH
#define KAAPI_ARCH_KCOMPUTERH 1

#if defined(__cplusplus)
extern "C" {
#endif


/*
type __sync_fetch_and_add (type *ptr, type value, ...)
type __sync_fetch_and_sub (type *ptr, type value, ...)
type __sync_fetch_and_or (type *ptr, type value, ...)
type __sync_fetch_and_and (type *ptr, type value, ...)
type __sync_fetch_and_xor (type *ptr, type value, ...)
type __sync_fetch_and_nand (type *ptr, type value, ...)
type __sync_add_and_fetch (type *ptr, type value, ...)
type __sync_sub_and_fetch (type *ptr, type value, ...)
type __sync_or_and_fetch (type *ptr, type value, ...)
type __sync_and_and_fetch (type *ptr, type value, ...)
type __sync_xor_and_fetch (type *ptr, type value, ...)
type __sync_nand_and_fetch (type *ptr, type value, ...)
bool __sync_bool_compare_and_swap (type *ptr, type oldval type newval, ...)
type __sync_val_compare_and_swap (type *ptr, type oldval type newval, ...)
*/

/* atomic compare and exchange. compare o with *ptr, if identical store
 * n in *ptr. return the initial value of *ptr.
 */
/* Heavily inspired by the Linux/asm/sparc code */
#define PROT_MASK8 (((1L << 32) - 1L) << 8)

static inline uint64_t __cmpxchg8(volatile uint8_t *ptr, uint8_t o, uint8_t n)
{
	uint32_t old,c,x;
	old = *ptr;
	c = ((o & ~PROT_MASK8) | (old & PROT_MASK8));
	x = ((n & ~PROT_MASK8) | (old & PROT_MASK8));
	__asm__ __volatile__ ("cas [%2], %3, %0"
				: "=&r" (x)
				: "r" (x), "r" (ptr), "r" (c)
				: "memory" );

	return (x & ~PROT_MASK8);
}

#define PROT_MASK16 (((1L << 32) - 1L) << 16)

static inline uint64_t __cmpxchg16(volatile uint16_t *ptr, uint16_t o, uint16_t n)
{
	uint32_t old,c,x;
	old = *ptr;
	c = ((o & ~PROT_MASK16) | (old & PROT_MASK16));
	x = ((n & ~PROT_MASK16) | (old & PROT_MASK16));
	__asm__ __volatile__ ("cas [%2], %3, %0"
				: "=&r" (x)
				: "r" (x), "r" (ptr), "r" (c)
				: "memory" );

	return (x & ~PROT_MASK16);
}


static inline uint64_t __cmpxchg32(volatile uint32_t *ptr, uint32_t o, uint32_t n)
{
	__asm__ __volatile__ ("cas [%2], %3, %0"
				: "=&r" (n)
				: "0" (n), "r" (ptr), "r" (o)
				: "memory" );
	return n;
}


static inline uint64_t __cmpxchg64(volatile uint64_t *ptr, uint64_t o, uint64_t n)
{
	__asm__ __volatile__ ("casx [%2], %3, %0"
				: "=&r" (n)
				: "0" (n), "r" (ptr), "r" (o)
				: "memory" );
	return n;
}

static inline uint64_t	__cmpxchg(volatile void *ptr, uint64_t o, uint64_t n, size_t sz)
{
	switch (sz)
	{
		case 1:
			return __cmpxchg8((volatile uint8_t *)ptr,o,n);
		case 2:
			return __cmpxchg16((volatile uint16_t *)ptr,o,n);
		case 4:
			return __cmpxchg32((volatile uint32_t *)ptr,o,n);
		case 8:
			return __cmpxchg64((volatile uint64_t *)ptr,o,n);
	}
	return o;
}

#define cmpxchg(ptr,o,n) 						\
	({								\
		__typeof__(*(ptr)) _o_ = (o);				\
		__typeof__(*(ptr)) _n_ = (n);				\
		(__typeof__(*(ptr))) __cmpxchg((ptr), (uint64_t)_o_,	\
		(uint64_t)_n_, sizeof(*(ptr))); 			\
	})

#define __sync_val_compare_and_swap(ptr,oldval,newval) 	cmpxchg(ptr,oldval,newval)
#define __sync_bool_compare_and_swap(ptr,oldval,newval) (cmpxchg(ptr,oldval,newval) == oldval)

// TODO: use the #StoreLoad and co
#define __sync_synchronize() do { __asm__ __volatile__ ("membar #StoreLoad" ::: "memory"); } while(0)

/*****************************
 * macro utilities
 ****************************/

#define CONCAT2x(a,b) a ## _ ## b
#define CONCAT2(a,b) CONCAT2x(a,b)
#define CONCAT3x(a,b,c) a ## _ ## b ## _ ## c
#define CONCAT3(a,b,c) CONCAT3x(a,b,c)
#define CONCTYPE(s) uint ## s ## _t

/*****************************
 * fetch_and_*
 ****************************/

#define __FETCHSOP(s,expr,name) 								\
static inline uint64_t CONCAT3(__sparcfa,name,s)(volatile CONCTYPE(s) *ptr, CONCTYPE(s) value)	\
{											\
	CONCTYPE(s) c, old;								\
	c = *ptr;									\
	for(;;) {									\
		old = cmpxchg(ptr,c,expr);						\
		if(old == c)								\
			break;								\
		c = old;								\
	}										\
	return c;									\
}

#define __FETCHOP(op,name)	\
	__FETCHSOP(8,op,name)	\
	__FETCHSOP(16,op,name)	\
	__FETCHSOP(32,op,name)	\
	__FETCHSOP(64,op,name)

__FETCHOP(c + value,add)
__FETCHOP(c - value,sub)
__FETCHOP(c | value,or)
__FETCHOP(c & value,and)
__FETCHOP(c ^ value,xor)
__FETCHOP(~c & value,nand)

#define __FETCHSW(name) 								\
static inline uint64_t	CONCAT2(__sparcfa,name)(volatile void *ptr, uint64_t v, size_t sz)	\
{											\
	switch (sz)									\
	{	 									\
		case 1:									\
			return CONCAT3(__sparcfa,name,8)((volatile CONCTYPE(8) *)ptr,v);	\
		case 2:									\
			return CONCAT3(__sparcfa,name,16)((volatile CONCTYPE(16) *)ptr,v); 	\
		case 4:									\
			return CONCAT3(__sparcfa,name,32)((volatile CONCTYPE(32) *)ptr,v); 	\
		case 8:									\
			return CONCAT3(__sparcfa,name,64)((volatile CONCTYPE(64) *)ptr,v);	\
	}										\
	return v;									\
}

__FETCHSW(add)
__FETCHSW(sub)
__FETCHSW(or)
__FETCHSW(and)
__FETCHSW(xor)
__FETCHSW(nand)

#define fetchname(ptr,v,name) 						\
	({								\
		__typeof__(*(ptr)) _v_ = (v);				\
		(__typeof__(*(ptr))) CONCAT2(__sparcfa,name)((ptr), (uint64_t)_v_,	\
		sizeof(*(ptr))); 			\
	})

#define __sync_fetch_and_add(ptr,value)		fetchname(ptr,value,add)
#define __sync_fetch_and_sub(ptr,value)		fetchname(ptr,value,sub)
#define __sync_fetch_and_or(ptr,value)		fetchname(ptr,value,or)
#define __sync_fetch_and_and(ptr, value)	fetchname(ptr,value,and)
#define __sync_fetch_and_xor(ptr, value)	fetchname(ptr,value,xor)
#define __sync_fetch_and_nand(ptr, value)	fetchname(ptr,value,nand)

/*****************************
 * *_and_fetch
 ****************************/

#define __OPSFETCH(s,expr,name) 								\
static inline uint64_t CONCAT3(__sparcaf,name,s)(volatile CONCTYPE(s) *ptr, CONCTYPE(s) value)	\
{											\
	CONCTYPE(s) c, old, v;								\
	c = *ptr;									\
	for(;;) {									\
		v = expr;								\
		old = cmpxchg(ptr,c,v);							\
		if(old == c)								\
			break;								\
		c = old;								\
	}										\
	return v;									\
}

#define __OPFETCH(op,name)	\
	__OPSFETCH(8,op,name)	\
	__OPSFETCH(16,op,name)	\
	__OPSFETCH(32,op,name)	\
	__OPSFETCH(64,op,name)

__OPFETCH(c + value,add)
__OPFETCH(c - value,sub)
__OPFETCH(c | value,or)
__OPFETCH(c & value,and)
__OPFETCH(c ^ value,xor)
__OPFETCH(~c & value,nand)

#define __OPSW(name) 								\
static inline uint64_t	CONCAT2(__sparcaf,name)(volatile void *ptr, uint64_t v, size_t sz)	\
{											\
	switch (sz)									\
	{	 									\
		case 1:									\
			return CONCAT3(__sparcaf,name,8)((volatile CONCTYPE(8) *)ptr,v); 		\
		case 2:									\
			return CONCAT3(__sparcaf,name,16)((volatile CONCTYPE(16) *)ptr,v); 	\
		case 4:									\
			return CONCAT3(__sparcaf,name,32)((volatile CONCTYPE(32) *)ptr,v); 	\
		case 8:									\
			return CONCAT3(__sparcaf,name,64)((volatile CONCTYPE(64) *)ptr,v); 	\
	}										\
	return v;									\
}

__OPSW(add)
__OPSW(sub)
__OPSW(or)
__OPSW(and)
__OPSW(xor)
__OPSW(nand)

#define opname(ptr,v,name) 						\
	({								\
		__typeof__(*(ptr)) _v_ = (v);				\
		(__typeof__(*(ptr))) CONCAT2(__sparcaf,name)((ptr), (uint64_t)_v_,	\
		sizeof(*(ptr))); 			\
	})

#define __sync_add_and_fetch(ptr,value)		opname(ptr,value,add)
#define __sync_sub_and_fetch(ptr,value)		opname(ptr,value,sub)
#define __sync_or_and_fetch(ptr,value)		opname(ptr,value,or)
#define __sync_and_and_fetch(ptr, value)	opname(ptr,value,and)
#define __sync_xor_and_fetch(ptr, value)	opname(ptr,value,xor)
#define __sync_nand_and_fetch(ptr, value)	opname(ptr,value,nand)

#if defined(__cplusplus)
}
#endif

#endif // defined KAAPI_ARCH_KCOMPUTER

#endif //defined __sparc_v9
