/*
** xkaapi
** 
**
** Copyright 2009,2010,2011,2012 INRIA.
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
#include "kaapi.h"

#define CONFIG_PAR_GRAIN 128

/* algorithm main function */
double accumulate(const double* beg, const double* end, double res)
{
  /* push an adaptive task */
  ka::StealContext* sc = ka::TaskBeginAdaptive(
        /* flag: concurrent which means concurrence between extrac_seq & splitter executions */
          KAAPI_SC_COOPERATIVE 
        /* flag: no preemption which means that not preemption will be available (few ressources) */
        | KAAPI_SC_NOPREEMPTION
  );

  for ( ; beg != end; ++beg)
  {
    ka::StealPoint( sc, 
        [&end, beg]( 
              int nreq,              /* number of requests */
              ka::Request* req       /* array of request */
        ) -> ka::Responder
        {
          size_t size = end - beg;
          if (size == 0) return;
          size_t req_size = size / (nreq + 1);
          if (req_size == 0)
          {
            nreq = size;
            req_size = 1;
          }
          double* end_theft = end;

          /* store new end ... */
          end -= nreq * req_size;
          
          /* here remains code may be executed in concurrence */
          for (; nreq; --nreq, ++req, end_theft -= unit_size)
            /* thief work: create a task */
            req->Spawn<TaskThief<T,OP> >(sc)( ka::pointer<T>(end_theft-req_size), ka::pointer<T>(end_theft), _op );
        }
    ); /* end steal point */
    res = res + *beg;
  }

  return res;
}


/* My main task */
struct doit {
  void operator()(int argc, char** argv )
  {
    size_t size = 10000;
    if (argc >1) size = atoi(argv[1]);
    
    double* array = new double[size];

    /* initialize, apply, check */
    memset(array, 0, sizeof(array));

    double res = accumulate( array, array+size, apply_sin );

    std::cout << "Done" << std::endl;
  }
};


/* main entry point : Kaapi initialization
*/
int main(int argc, char** argv)
{
  try {
    /* Join the initial group of computation : it is defining
       when launching the program by a1run.
    */
    ka::Community com = ka::System::join_community( argc, argv );
    
    /* Start computation by forking the main task */
    ka::SpawnMain<doit>()(argc, argv); 
    
    /* Leave the community: at return to this call no more athapascan
       tasks or shared could be created.
    */
    com.leave();

    /* */
    ka::System::terminate();
  }
  catch (const std::exception& E) {
    ka::logfile() << "Catch : " << E.what() << std::endl;
  }
  catch (...) {
    ka::logfile() << "Catch unknown exception: " << std::endl;
  }
  
  return 0;
}
