/****************************************************************************
 * 
 *  Test spawn of task
 *
 ***************************************************************************/
#ifndef _TEST_TASK_H
#define _TEST_TASK_H
#include <iostream>
#include "kaapi++" // this is the new C++ interface for Kaapi

template<class T>
struct TaskPrint : public ka::Task<3>::Signature<ka::R<T>, const char*, const T&> {};

/* */
template<class T>
struct TaskW : public ka::Task<1>::Signature<ka::W<T> > {};

template<class T>
struct TaskR : public ka::Task<1>::Signature<ka::R<T> > {};

template<class T>
struct TaskRW : public ka::Task<1>::Signature<ka::RW<T> > {};

template<class T>
struct TaskWp : public ka::Task<1>::Signature<ka::WP<T> > {};

template<class T>
struct TaskRp : public ka::Task<1>::Signature<ka::RP<T> > {};

template<class T>
struct TaskRpWp : public ka::Task<1>::Signature<ka::RPWP<T> > {};


template<class T>
struct TaskBodyCPU<TaskW<T> >  {
  void operator() ( ka::pointer_w<T> x )
  { }
};


template<class T>
struct TaskBodyCPU<TaskR<T> >  {
  void operator() ( ka::pointer_r<T> x )
  { }
};


template<class T>
struct TaskBodyCPU<TaskRW<T> >  {
  void operator() ( ka::pointer_rw<T> x )
  { }
};


template<class T>
struct TaskBodyCPU<TaskWp<T> >  {
  void operator() ( ka::pointer_wp<T> x )
  { }
};

template<class T>
struct TaskBodyCPU<TaskRp<T> >  {
  void operator() ( ka::pointer_rp<T> x )
  { }
};


template<class T>
struct TaskBodyCPU<TaskRpWp<T> >  {
  void operator() ( ka::pointer_rpwp<T> x )
  { }
};

#endif