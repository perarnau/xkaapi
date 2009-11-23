/* KAAPI public interface */
// =========================================================================
// (c) INRIA, projet MOAIS, 2009
// Author: T. Gautier, X-Kaapi port
//
//
//
// =========================================================================
#ifndef _ATHAPASCAN_1_H_H
#define _ATHAPASCAN_1_H_H

// This is the new version on top of X-Kaapi
extern "C" { const char* get_kaapi_version(); }

#include "kaapi.h"
#include <vector>

namespace a1 {

  using ::operator<<;
  using ::operator>>;

 enum { STACK_ALLOC_THRESHOLD = DFG::STACK_ALLOC_THRESHOLD }; 

  // --------------------------------------------------------------------
  namespace FormatDef {
    using namespace Util::FormatDef;
  }
  class Format;
  class UpdateFunctionFormat;

  class IStream;
  class OStream;
  class Exception;
  class RuntimeError;
  class InvalidArgumentError;
  class RestartException;
  class ServerException;
  class NoFound;
  class BadAlloc;
  class IOError;
  class ComFailure;
  class BadURL;

  // --------------------------------------------------------------------
  class Community {
  protected:
    friend class System;
    Community( void* com )
    { }

  public:
    Community( const Community& com );

    /* */
    void commit();

    /* */
    void leave();

    /* */
    bool is_leader() const;
  };

  // --------------------------------------------------------------------
  struct Thread;
  
  // --------------------------------------------------------------------
  class System {
  public:
    static Community join_community( int& argc, char**& argv )
      throw (RuntimeError,RestartException,ServerException);

    static Community initialize_community( int& argc, char**& argv )
      throw (RuntimeError,RestartException,ServerException);

    static Thread* get_current_thread();
    static int getRank();
    static void terminate();

  public:
  };

  // --------------------------------------------------------------------
  inline Thread* System::get_current_thread()
  {
    return (Thread*)kaapi_self_stack();
  }

  // --------------------------------------------------------------------
  struct SetStack {};
  extern SetStack SetInStack;

  // --------------------------------------------------------------------
  struct SetHeap {};
  extern SetHeap SetInHeap;

  // --------------------------------------------------------------------
  class SetStickyC{};
  extern SetStickyC SetSticky;


  // --------------------------------------------------------------------
  template<class T>
  class Shared {
  public:
    typedef T value_type;

    ~Shared ( ) 
    {
      Thread* thread = System::get_current_thread(); 
      destroy( thread );
    }
    
    Shared ( value_type* data = 0 ) 
    {
      Thread* thread = System::get_current_thread(); 
      if (!data) 
      {
        data = 0;
        if (sizeof(value_type) <= STACK_ALLOC_THRESHOLD) 
        {
            attr.set_instack();
            data = new (thread->allocate(sizeof(value_type))) value_type;
        } else {
            attr.set_inheap();
#if defined(KAAPI_USE_NUMA)
              //WARN LAURENT : hack to sched
              data = new value_type;
#endif
          }
      }
      else
      {
          attr.set_inheap();
      }
      initialize( thread, data, &Util::WrapperFormat<value_type>::theformat, attr);
    }

    Shared ( const SetStack& toto, value_type* data = 0) 
    {
      Thread* thread = System::get_current_thread(); 
      DFG::GlobalData::Attribut attr;
      attr.clear();
      attr.set_instack();
      if(!data) data = new (thread->allocate(sizeof(value_type))) value_type;
      initialize( thread, data, &Util::WrapperFormat<value_type>::theformat, attr);
    }

    Shared ( const SetHeap& toto, value_type* data = 0) 
    {
      Thread* thread = System::get_current_thread(); 
      DFG::GlobalData::Attribut attr;
      attr.clear();
      attr.set_inheap();
      if(!data) data =
#if defined(KAAPI_USE_NUMA)
      //WARN LAURENT
        new value_type;
#else
        0;
#endif
      initialize( thread, data, &Util::WrapperFormat<value_type>::theformat, attr);
    }

    Shared(const value_type& value )
    {
      if (sizeof(value_type) <= STACK_ALLOC_THRESHOLD) {
        Thread* thread = System::get_current_thread(); 
        _gd.data  = new (SharedAllocator, thread) T(value)
        _gd._attr = 0;
      } else {
        _gd.data  = new (SharedAllocator) T(value)
        _gd._attr = 1;
      }
    }

    Shared(const SetStack& toto, const T& value )
    {
      Thread* thread = System::get_current_thread(); 
      _gd.data  = new (SharedAllocator, thread) T(value)
      _gd._attr = 0;
    }

    Shared(const SetHeap& toto, const T& value )
    {
      _gd.data  = new (SharedAllocator) T(value)
      _gd._attr = 1;
    }

    Shared(const Shared<value_type>& t) 
     : _gd(t._gd)
    {
      t._gd.data    = 0;
      t._gd.version = 0;
    }

    Shared<T>& operator=(const Shared<value_type>& t) 
    {
      _gd = t._gd;
      t._gd.data    = 0;
      t._gd.version = 0;
      return *this;
    }

  private:
    kaapi_access_t _gd;
    int            _attr; /* 0: in stack, 1: in heap */
  };
  

  // --------------------------------------------------------------------
  template<class T>
  class Shared_rp {
  public:
    typedef T value_type;

    Shared_rp( const kaapi_access_t& a )
     : _gd( a )
    { }
  protected:
    kaapi_access_t _gd;
  };


  // --------------------------------------------------------------------
  template<class T>
  class Shared_r : public DFG::Shared_r {
  public:
    typedef T value_type;

    Shared_r( const kaapi_access_t& a )
     : _gd( a )
    { }

    const value_type& read() const 
    { return *_gd.data; }
  protected:
    kaapi_access_t _gd;
  };


  // --------------------------------------------------------------------
  template<class T>
  class Shared_wp  {
  public:
    typedef T value_type;

    Shared_wp( const kaapi_access_t& a )
     : _gd( a )
    { }
  protected:
    kaapi_access_t _gd;
  };


  // --------------------------------------------------------------------
  template<class T>
  class Shared_w {
  public:
    typedef T value_type;

    Shared_w( const kaapi_access_t& a )
     : _gd( a )
    { }

    void write( const value_type& new_value )
    { 
      T* data = (T*)_gd.data;
      *data = new_value;
    }

    void write(value_type* new_value) 
    { 
      T* data = (T*)_gd.data;
      *data = *new_value;
    }
  protected:
    kaapi_access_t _gd;
  };


  // --------------------------------------------------------------------
  template<class T>
  class Shared_rpwp {
  public:
    typedef T value_type;

    Shared_rpwp( const kaapi_access_t& a )
     : _gd( a )
    { }
  protected:
    kaapi_access_t _gd;
  };


  // --------------------------------------------------------------------
  template<class T>
  class Shared_rw {
  public:
    typedef T value_type;

    Shared_rw( const kaapi_access_t& a )
     : _gd( a )
    { }

    value_type& access() const
    { return *_gd.data; }
  protected:
    kaapi_access_t _gd;
  };


  // --------------------------------------------------------------------
  template<class T>
  struct DefaultAdd {
    void operator()( T& result, const T& value ) const
    {
      result += value;
    }
  };
  
  template<class T, class OpCumul = DefaultAdd<T> >
  class Shared_cwp {
  public:    
    typedef T value_type;

    Shared_cwp(const kaapi_access_t& a )
     : _gd( a )
    { }
  };


  template<class T, class OpCumul = DefaultAdd<T> >
  class Shared_cw {
  public:
    typedef T value_type;

    Shared_cw( const kaapi_access_t& a )
     : _gd( a )
    { }

    void cumul( const value_type& value )
    {
      static OpCumul op;
      op( *_gd._data, value );
    }

    void cumul( value_type* value )
    { 
      op( *_gd._data, *value );
      delete value;
    }
  };


  // --------------------------------------------------------------------
  /* used to report information about shared in order to simply Trait_Link and Trait_type classes */
  template<class T> 
  class Trait_Shared {
  public:
    enum { isshared = false };
    enum { mode_val = KAAPI_ACCESS_MODE_V };
    typedef T type_inclosure;
    typedef T type_f;
  };
  
  template<class T>
  class Trait_Shared<Shared<T> > {
  public:
    enum { isshared = true };
    enum { mode_val = KAAPI_ACCESS_MODE_RW|KAAPI_ACCESS_MODE_P };
    typedef Shared_cannot_be_in_parameter type_inclosure;
    typedef Shared_cannot_be_in_parameter type_f;
  };

  template<class T>
  class Trait_Shared<Shared_r<T> > {
  public:
    enum { isshared = true };
    enum { mode_val = KAAPI_ACCESS_MODE_R };
    typedef kaapi_access_t type_inclosure;
    typedef Shared_r<T> type_f;
  };

  template<class T>
  class Trait_Shared<Shared_rp<T> > {
  public:
    enum { isshared = true };
    enum { mode_val = KAAPI_ACCESS_MODE_R|KAAPI_ACCESS_MODE_P };
    typedef kaapi_access_t type_inclosure;
    typedef Shared_rp<T> type_f;
  };

  template<class T>
  class Trait_Shared<Shared_w<T> > {
  public:
    enum { isshared = true };
    enum { mode_val = KAAPI_ACCESS_MODE_W };
    typedef kaapi_access_t type_inclosure;
    typedef Shared_w<T> type_f;
  };
  template<class T>
  class Trait_Shared<Shared_wp<T> > {
  public:
    enum { isshared = true };
    enum { mode_val = KAAPI_ACCESS_MODE_W|KAAPI_ACCESS_MODE_P };
    typedef kaapi_access_t type_inclosure;
    typedef Shared_wp<T> type_f;
  };
  

  // --------------------------------------------------------------------
  /* used to report constraint in the mapping of tasks depending of the parameter */
  template<class F> 
  class Trait_Constraint {
  public:
    enum { local = false };
  };
  

  // -------------------------------------------------------------------- VECTOR of Shared
//\TODO


  // -------------------------------------------------------------------- VECTOR of Shared


  // --------------------------------------------------------------------
  typedef kaapi_task_t Closure;
  
  class DefaultAttribut {
  public:
    Closure* operator()( Thread*, Closure* clo)
    { return clo; }
  };
  extern DefaultAttribut SetDefault;
  
  /* */
  class UnStealableAttribut {
  public:
    Closure* operator()( Thread*, Closure* clo)
    { clo->flag |= KAAPI_TASK_STICKY; return clo; }
  };
  inline UnStealableAttribut SetUnStealable()
  { return UnStealableAttribut(); }

  /* like default attribut: not yet distributed computation */
  class SetLocalAttribut {
  public:
    Closure* operator()( Thread*, Closure* clo)
    {  return clo; }
  };
  extern SetLocalAttribut SetLocal;

  /* DEPRECATED??? to nothing... not yet distributed implementation */
  class AttributSetCost {
    float _cost;
  public:
    AttributSetCost( float c ) : _cost(c) {}
    Closure* operator()( Thread*, Closure* clo)
    { return clo; }
  };
  inline AttributSetCost SetCost( float c )
  { return AttributSetCost(c); }


  /* to nothing... not yet distributed implementation */
  class AttributSetSite {
    int _site;
  public:
    AttributSetSite( int s ) : _site(s) {}
    Closure* operator()( Thread*, Closure* clo)
    { return clo; }
  };

  inline AttributSetSite SetSite( int s )
  { return AttributSetSite(s); }
  

  // \TODO at the end (...)
#if defined(KAAPI_USE_ST)
  class SetStaticSchedAttribut {
    int _npart;
    int _niter;
    Sched::PartitionTask::GraphType _type;
  public:
    SetStaticSchedAttribut( int n, int m,Sched::PartitionTask::GraphType type=Sched::PartitionTask::STANDARD ) 
     : _npart(n), _niter(m),_type(type) {}
    template<class A1_CLO>
    RFO::Closure* operator()( Thread* t, A1_CLO*& clo)
    { 
      /* Replace Clo by an other task */
      Sched::PartitionTask* sched_clo = new (t->allocate(sizeof(Sched::PartitionTask))) Sched::PartitionTask;
      sched_clo->initialize();
      sched_clo->clear();
      sched_clo->set_local();
      sched_clo->unset_stealable();
      sched_clo->set_format( &Sched::ClosureFormatPartitionTask::theformat );
      sched_clo->set_run( Sched::ClosureFormatPartitionTask::theformat.get_run() );
      sched_clo->run_noexec = &A1_CLO::srun_onelevel;
      sched_clo->run_internal = &A1_CLO::srun_internal;
      sched_clo->run_release  = &A1_CLO::srun_release;
      sched_clo->orig_clo   = clo;
      sched_clo->npart      = _npart;
      sched_clo->type       = _type;
      sched_clo->niter      = _niter;
      sched_clo->curr_iter  = 0;
      sched_clo->tgid       = Sched::ThreadGroup::UNDEF_THREADGROUP;

      clo = (A1_CLO*)sched_clo; /* only for typechecking */
      return sched_clo; 
    }
  };
  inline SetStaticSchedAttribut SetStaticSched(int npart, int iter = 1 ,Sched::PartitionTask::GraphType type=Sched::PartitionTask::STANDARD)
  { return SetStaticSchedAttribut(npart, iter,type); }
#endif

#if 0 //\TODO dans un monde ideal, il faudrait ca
#include "atha_spacecollection.h"
#endif
#include "athapascan_closure.h"

  // --------------------------------------------------------------------
  /* New API: thread.Fork<Task>([ATTR])( args )
     Fork<Task>([ATTR])(args) with be implemented on top of 
     System::get_current_thread()->Fork<Task>([ATTR])( args ).
  */
  class Thread {
    kaapi_stack_t _stack;
  public:
    template<class TASK,class ATTRIBUT>
    class Forker {
      Thread*              _thread;
      ATTRIBUT             _attr;
    public:
      Forker(Thread* t, const ATTRIBUT& attr) 
       : _thread(t), _attr(attr)
      { }

    public:
      /* case 0 was not automatically generated */
      Closure* operator()( )
      {
        kaapi_task_t* clo = kaapi_stack_toptask( &_stack);
        kaapi_task_init( &_stack, clo, KaapiClosure0<TASK>::body, 0 );
        _attr(_thread, clo);
        kaapi_stack_pushtask( &_stack);    
        return clo;
      }

#      include "athapascan_fork.h"

      };
      
      template<class TASK>
      Forker<TASK,DefaultAttribut> Fork()
      { 
        return Forker<TASK,DefaultAttribut>(this,SetDefault);
      }

      template<class TASK, class ATTR>
      Forker<TASK,ATTR> Fork(const ATTR& attr)
      { 
        return Forker<TASK,ATTR>(this, attr);
      }

  }; /* Thread */

  template<class TASK>
  class MainTask {};
  
  template<class TASK>
  struct ForkerMain : protected Forker<MainTask<TASK>,DefaultAttribut> 
  {
    ForkerMain() 
     : Forker<MainTask<TASK>,DefaultAttribut>(0,SetDefault)
    { }

    void operator()( int argc, char** argv)
    {
      TASK()( argc, argv );
    }
  };

  template<class TASK>
  ForkerMain<TASK> ForkMain()
  { 
    return ForkerMain<TASK>();
  }
  
  /** Wait execution of all forked tasks of the running task */
  extern void Sync();

#if 0
  /** Private interface ??? */
  template<class TASK>
  inline const ClosureFormat* GetClosureFormat( )
  {
    return HandlerGetClosureFormat( &TASK::operator() );
  }
#endif

} // namespace a1

#if 0 // \TODO à reprendre ...
template<class T, class F>
inline a1::OStream& operator<<( a1::OStream& s_out, const a1::GlobalVariable<T,F>& c)
{ return s_out << c._gv; }

template<class T, class F>
inline a1::IStream& operator>>( a1::IStream& s_in, a1::GlobalVariable<T,F>& c)
{ return s_in >> c._gv; }

template<class T, class F>
inline a1::OStream& operator<<( a1::OStream& s_out, const a1::MonotonicBound<T,F>& c)
{ return s_out << c._gv; }

template<class T, class F>
inline a1::IStream& operator>>( a1::IStream& s_in, a1::MonotonicBound<T,F>& c)
{ return s_in >> c._gv; }

template<class T>
inline a1::OStream& operator<<( a1::OStream& s_out, const a1::SingleAssignment<T>& c)
{ return s_out << c._gv; }

template<class T>
inline a1::IStream& operator>>( a1::IStream& s_in, a1::SingleAssignment<T>& c)
{ return s_in >> c._gv; }

template<class T>
inline a1::OStream& operator<<( a1::OStream& s_out, const a1::Shared<T>& /*c*/)
{ return s_out; }

template<class T>
inline a1::IStream& operator>>( a1::IStream& s_in, a1::Shared<T>& /*c*/)
{ return s_in; }

template<class T>
inline a1::OStream& operator<<( a1::OStream& s_out, const a1::Shared_r<T>& /*c*/)
{ return s_out; }

template<class T>
inline a1::IStream& operator>>( a1::IStream& s_in, a1::Shared_r<T>& /*c*/)
{ return s_in; }
#endif

#if 0 // \TODO a specifier...
template<class T>
inline a1::OStream& operator<<( a1::OStream& s_out, const a1::SpaceCollection<T>& /*c*/)
{ return s_out; }

template<class T>
inline a1::IStream& operator>>( a1::IStream& s_in, a1::SpaceCollection<T>& /*c*/)
{ return s_in; }

template<class T>
inline a1::OStream& operator<<( a1::OStream& s_out, const a1::SpaceCollection_rw<T>& /*c*/)
{ return s_out; }

template<class T>
inline a1::IStream& operator>>( a1::IStream& s_in, a1::SpaceCollection_rw<T>& /*c*/)
{ return s_in; }

inline a1::OStream& operator<<( a1::OStream& s_out, const a1::SpaceCollectionRep& /*c*/)
{ return s_out; }

inline a1::IStream& operator>>( a1::IStream& s_in, a1::SpaceCollectionRep& /*c*/)
{ return s_in; }
#endif


#if 0 // \TODO: a reprendre correctement
/* specialize the resize of vector of shared */
namespace a1 {
template<class T, class Alloc>
inline void resize_vector(std::vector<a1::Shared<T>,Alloc>& v, typename std::vector<a1::Shared<T>,Alloc>::size_type __new_size)
{
  typename std::vector<a1::Shared<T>,Alloc>::size_type sz = v.size();
  v.resize(__new_size);
  for (typename std::vector<a1::Shared<T>,Alloc>::size_type i=sz; i<__new_size; ++i)
    v[i] = Shared<T>();
}
}
#endif


// ---------------------------------------------------------------------------------
namespace a1 {
  class SyncGuard {
      Thread       *_thread;
      kaapi_frame_t _frame;
  public:
      SyncGuard() : _thread( System::get_current_thread() )
      {
        kaapi_stack_save_frame( &_thread->_stack, &_frame );
      }
      ~SyncGuard()
      {
        // \TODO: only execution in one frame and sub frame but not on top frame
        kaapi_stack_restore_frame( &_thread->_stack, &_frame );
      }
  };
}

#ifndef ATHAPASCAN_NOT_IN_NAMESPACE
using namespace a1;
#endif

#endif
