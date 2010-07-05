#ifndef CONFIG_HH_INCLUDED
# define CONFIG_HH_INCLUDED



// compile time config. overridable.

// available algorithms
#ifndef CONFIG_ALGO_FOR_EACH
#define CONFIG_ALGO_FOR_EACH 0
#endif

#ifndef CONFIG_ALGO_COUNT
#define CONFIG_ALGO_COUNT 0
#endif

#ifndef CONFIG_ALGO_SEARCH
#define CONFIG_ALGO_SEARCH 0
#endif

#ifndef CONFIG_ALGO_ACCUMULATE
#define CONFIG_ALGO_ACCUMULATE 0
#endif

#ifndef CONFIG_ALGO_TRANSFORM
#define CONFIG_ALGO_TRANSFORM 0
#endif

#ifndef CONFIG_ALGO_MIN_ELEMENT
#define CONFIG_ALGO_MIN_ELEMENT 0
#endif

#ifndef CONFIG_ALGO_MAX_ELEMENT
#define CONFIG_ALGO_MAX_ELEMENT 0
#endif

#ifndef CONFIG_ALGO_FIND
#define CONFIG_ALGO_FIND 0
#endif

#ifndef CONFIG_ALGO_FIND_IF
#define CONFIG_ALGO_FIND_IF 0
#endif

#ifndef CONFIG_ALGO_SWAP_RANGES
#define CONFIG_ALGO_SWAP_RANGES 0
#endif

#ifndef CONFIG_ALGO_INNER_PRODUCT
#define CONFIG_ALGO_INNER_PRODUCT 0
#endif

// lib to use
#ifndef CONFIG_LIB_TBB
#define CONFIG_LIB_TBB 0
#endif

#ifndef CONFIG_LIB_KASTL
#define CONFIG_LIB_KASTL 0
#endif

#ifndef CONFIG_LIB_STL
#define CONFIG_LIB_STL 0
#endif

#ifndef CONFIG_LIB_PASTL
#define CONFIG_LIB_PASTL 0
#endif


// action to do. exclusive.
#ifndef CONFIG_DO_CHECK
#define CONFIG_DO_CHECK 0
#endif

#ifndef CONFIG_DO_BENCH
#define CONFIG_DO_BENCH 0
#endif



#endif // ! CONFIG_HH_INCLUDED
