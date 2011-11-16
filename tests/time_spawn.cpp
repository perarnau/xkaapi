/*
** xkaapi
** 
**
** Copyright 2009 INRIA.
**
** Contributors :
**
** thierry.gautier@inrialpes.fr
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
#include "test_main.h"
#include "kaapi++"


/* null task */
struct TaskArg0 : public ka::Task<0>::Signature {};
template<>
struct TaskBodyCPU<TaskArg0>  {
  void operator() ( )
  { }
};


/* task with 1 arg */
struct TaskArg1 : public ka::Task<1>::Signature<ka::RPWP<int> > {};
template<>
struct TaskBodyCPU<TaskArg1>  {
  void operator() ( ka::pointer_rpwp<int> x )
  { }
};

/* task with 2 arg */
struct TaskArg2 : public ka::Task<2>::Signature<ka::RPWP<int>, ka::RPWP<int> > {};
template<>
struct TaskBodyCPU<TaskArg2>  {
  void operator() ( ka::pointer_rpwp<int> x1, ka::pointer_rpwp<int> x2 )
  { }
};


/* task with 4 arg */
struct TaskArg4 : public ka::Task<4>::Signature<ka::RPWP<int>, ka::RPWP<int>, ka::RPWP<int>, ka::RPWP<int> > {};
template<>
struct TaskBodyCPU<TaskArg4>  {
  void operator() ( 
    ka::pointer_rpwp<int> x1, ka::pointer_rpwp<int> x2,
    ka::pointer_rpwp<int> x3, ka::pointer_rpwp<int> x4
  )
  { }
};


/* task with 5 arg */
struct TaskArg5 : public ka::Task<5>::Signature<ka::RPWP<int>, ka::RPWP<int>, ka::RPWP<int>, ka::RPWP<int>, ka::RPWP<int> > {};
template<>
struct TaskBodyCPU<TaskArg5>  {
  void operator() ( 
    ka::pointer_rpwp<int> x1, ka::pointer_rpwp<int> x2,
    ka::pointer_rpwp<int> x3, ka::pointer_rpwp<int> x4,
    ka::pointer_rpwp<int> x5
  )
  { }
};

/* task with 6 arg */
struct TaskArg6 : public ka::Task<6>::Signature<ka::RPWP<int>, ka::RPWP<int>, ka::RPWP<int>, ka::RPWP<int>, ka::RPWP<int>, ka::RPWP<int> > {};
template<>
struct TaskBodyCPU<TaskArg6>  {
  void operator() ( 
    ka::pointer_rpwp<int> x1, ka::pointer_rpwp<int> x2,
    ka::pointer_rpwp<int> x3, ka::pointer_rpwp<int> x4,
    ka::pointer_rpwp<int> x5, ka::pointer_rpwp<int> x6
  )
  { }
};


/* task with 8 arg */
struct TaskArg8 : public ka::Task<8>::Signature<ka::RPWP<int>, ka::RPWP<int>, ka::RPWP<int>, ka::RPWP<int>, ka::RPWP<int>, ka::RPWP<int>, ka::RPWP<int>, ka::RPWP<int> > {};
template<>
struct TaskBodyCPU<TaskArg8>  {
  void operator() ( 
    ka::pointer_rpwp<int> x1, ka::pointer_rpwp<int> x2,
    ka::pointer_rpwp<int> x3, ka::pointer_rpwp<int> x4,
    ka::pointer_rpwp<int> x5, ka::pointer_rpwp<int> x6,
    ka::pointer_rpwp<int> x7, ka::pointer_rpwp<int> x8
  )
  { }
};



#define DO_EXP(argcount,spawninst) \
  t0 = kaapi_get_elapsedns();\
  for (int k=0; k<iter; ++k)\
  {\
    for (int i=0; i<ntasks; ++i)\
      ka::spawninst;\
    kaapi_thread_restore_frame((kaapi_thread_t*)thread, &frame );\
  }\
  t1 = kaapi_get_elapsedns();\
  std::cout << argcount << "\t" << (t1 - t0)/(double)(ntasks*iter) << std::endl;\


#define DO_EXP_CTXT(argcount,spawninst) \
  t0 = kaapi_get_elapsedns();\
  for (int k=0; k<iter; ++k)\
  {\
    for (int i=0; i<ntasks; ++i)\
      thread->spawninst;\
    kaapi_thread_restore_frame((kaapi_thread_t*)thread, &frame );\
  }\
  t1 = kaapi_get_elapsedns();\
  std::cout << argcount << "\t" << (t1 - t0)/(double)(ntasks*iter) << std::endl;\



/* Main of the program
*/
void doit::operator()(int argc, char** argv )
{
  int ntasks = atoi(argv[1]);
  double t0, t1;
  int x1,x2,x3,x4,x5,x6,x7,x8;
  int iter = 1000;
  kaapi_frame_t frame;
  ka::Thread* thread = ka::System::get_current_thread();
  kaapi_thread_save_frame( (kaapi_thread_t*)thread, &frame );

  /* warmup */
  for (int k=0; k<iter; ++k)
  {
    for (int i=0; i<ntasks; ++i)
      thread->Spawn<TaskArg8>()(&x1,&x2,&x3,&x4,&x5,&x6,&x7,&x8);
    kaapi_thread_restore_frame((kaapi_thread_t*)thread, &frame );
  }

  std::cout << "\n# Time ka::Spawn" << std::endl;
  std::cout << "# #args\t  Time(ns)" << std::endl;
  DO_EXP(0,Spawn<TaskArg0>()());
  DO_EXP(1,Spawn<TaskArg1>()(&x1));
  DO_EXP(2,Spawn<TaskArg2>()(&x1,&x2));
  DO_EXP(4,Spawn<TaskArg4>()(&x1,&x2,&x3,&x4));
  DO_EXP(5,Spawn<TaskArg5>()(&x1,&x2,&x3,&x4,&x5));
  DO_EXP(6,Spawn<TaskArg6>()(&x1,&x2,&x3,&x4,&x5,&x6));
  DO_EXP(8,Spawn<TaskArg8>()(&x1,&x2,&x3,&x4,&x5,&x6,&x7,&x8));

  std::cout << "\n# Time thread->Spawn" << std::endl;
  std::cout << "# #args\t  Time(ns)" << std::endl;
  DO_EXP_CTXT(0,Spawn<TaskArg0>()());
  DO_EXP_CTXT(1,Spawn<TaskArg1>()(&x1));
  DO_EXP_CTXT(2,Spawn<TaskArg2>()(&x1,&x2));
  DO_EXP_CTXT(4,Spawn<TaskArg4>()(&x1,&x2,&x3,&x4));
  DO_EXP_CTXT(5,Spawn<TaskArg5>()(&x1,&x2,&x3,&x4,&x5));
  DO_EXP_CTXT(6,Spawn<TaskArg6>()(&x1,&x2,&x3,&x4,&x5,&x6));
  DO_EXP_CTXT(8,Spawn<TaskArg8>()(&x1,&x2,&x3,&x4,&x5,&x6,&x7,&x8));
}
