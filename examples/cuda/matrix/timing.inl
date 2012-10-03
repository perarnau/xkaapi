
#if defined(KAAPI_TIMING)

#define	    KAAPI_TIMING_BEGIN() \
    double _t0 = kaapi_get_elapsedtime()

#define	    KAAPI_TIMING_END(task,n)			    \
    do{							    \
	double _t1 = kaapi_get_elapsedtime();		    \
	fprintf(stdout, "%s %d %.10f\n", task, n, _t1-_t0);   \
    }while(0)

#if defined(CONFIG_USE_CUDA)

#define	    KAAPI_TIMING_CUDA_BEGIN(stream)	\
    cudaEvent_t _evt0,_evt1;			\
    do{						\
        cudaEventCreate(&_evt0);		\
        cudaEventCreate(&_evt1);		\
        cudaEventRecord(_evt0,stream);		\
    }while(0)

#define	    KAAPI_TIMING_CUDA_END(stream,task,n)	    \
    do{							    \
	cudaEventRecord(_evt1,stream);			    \
	cudaEventSynchronize(_evt1);			    \
	float _tdelta;					    \
	cudaEventElapsedTime(&_tdelta, _evt0, _evt1);	    \
	fprintf(stdout, "%s %d %.10f\n", task, n, (_tdelta/1e3) );\
    }while(0)

#endif /* CONFIG_USE_CUDA */

#else /* KAAPI_TIMING */

#define	    KAAPI_TIMING_BEGIN()
#define	    KAAPI_TIMING_END(task,n)
#define	    KAAPI_TIMING_CUDA_BEGIN(stream)
#define	    KAAPI_TIMING_CUDA_END(stream,task,n)

#endif /* KAAPI_USE_TIMING */

