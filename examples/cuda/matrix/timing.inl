/*
 ** xkaapi
 **
 ** Copyright 2009, 2010, 2011, 2012 INRIA.
 **
 ** Contributors :
 **
 ** thierry.gautier@inrialpes.fr
 ** Joao.Lima@imag.fr / joao.lima@inf.ufrgs.br
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

