// --------------------------------------------------------------------
/* KAAPI_NUMBER_PARAMS is the number of possible parameters */
template<class TASK M4_PARAM(`,class F$1', `', ` ')>
class KAAPI_CLOSURE(KAAPI_NUMBER_PARAMS) : public RFO::Closure { 
public:
 M4_PARAM(`typename Trait_Type<F$1>::type_inclosure f$1;
  ', ` ', `')
 M4_PARAM(`typedef typename Trait_Type<F$1>::type_f type_f$1;
  ', ` ', `')
  static void srun(RFO::Closure* c, Core::Thread* t)
  {
    static TASK dummy;
    typedef KAAPI_CLOSURE(KAAPI_NUMBER_PARAMS)<TASK M4_PARAM(`,F$1', `', `')> TheClosure;
    TheClosure* clo = (TheClosure*)c;
    Thread* thread = (Thread*)t;
    KAAPI_ASSERT_DEBUG_M( thread == System::get_current_thread(), "bad assertion");

    RFO::Frame frame;
    thread->push( &frame );
    dummy(M4_PARAM(`clo->f$1', `', `, '));
    /* inline release code : clo->get_format()->release(clo) */
    M4_PARAM(`Trait_Type<F$1>::release(&clo->f$1);
    ', `', `', `')
    if (!frame.empty()) thread->execute( &frame );
    clo->set_state(RFO::Closure::S_TERM);
    thread->pop();
  }

  static void srun_onelevel(RFO::Closure* c, Core::Thread* t )
  {
    static TASK dummy;
    typedef KAAPI_CLOSURE(KAAPI_NUMBER_PARAMS)<TASK M4_PARAM(`,F$1', `', `')> TheClosure;
    TheClosure* clo = (TheClosure*)c;
    KAAPI_ASSERT_DEBUG_M( ((Thread*)t) == System::get_current_thread(), "bad assertion");

    dummy(M4_PARAM(`clo->f$1', `', `, '));
    /* inline release code : clo->get_format()->release(clo) */
    M4_PARAM(`Trait_Type<F$1>::release(&clo->f$1);
    ', `', `', `')
    clo->set_state(RFO::Closure::S_TERM);
  }

  static void srun_internal(RFO::Closure* c, Core::Thread* t )
  {
    static TASK dummy;
    typedef KAAPI_CLOSURE(KAAPI_NUMBER_PARAMS)<TASK M4_PARAM(`,F$1', `', `')> TheClosure;
    TheClosure* clo = (TheClosure*)c;
    KAAPI_ASSERT_DEBUG_M( ((Thread*)t) == System::get_current_thread(), "bad assertion");

    dummy(M4_PARAM(`clo->f$1', `', `, '));
  }
  
  static void srun_release(RFO::Closure* c, Core::Thread* t )
  {
    typedef KAAPI_CLOSURE(KAAPI_NUMBER_PARAMS)<TASK M4_PARAM(`,F$1', `', `')> TheClosure;
    TheClosure* clo = (TheClosure*)c;
    KAAPI_ASSERT_DEBUG_M( ((Thread*)t) == System::get_current_thread(), "bad assertion");
    /* inline release code : clo->get_format()->release(clo) */
    M4_PARAM(`Trait_Type<F$1>::release(&clo->f$1);
    ', `', `', `')
    clo->set_state(RFO::Closure::S_TERM);
  }
  
  static void srun_steal(RFO::Closure* c, Core::Thread* t)
  {
    static TASK dummy;
    typedef KAAPI_CLOSURE(KAAPI_NUMBER_PARAMS)<TASK M4_PARAM(`,F$1', `', `')> TheClosure;
    TheClosure* clo __attribute__((unused)) = (TheClosure*)c;
    Thread* thread = (Thread*)t;
    KAAPI_ASSERT_DEBUG_M( thread == System::get_current_thread(), "bad assertion");

    RFO::Frame frame;
    thread->push( &frame );
    dummy(M4_PARAM(`clo->f$1', `', `, '));
    /* inline release code : clo->get_format()->release(clo) */
    if (!frame.empty()) thread->execute( &frame );
    M4_PARAM(`Trait_Type<F$1>::release(&clo->f$1);
    ', `', `', `')
    thread->pop();
  }

};



// --------------------------------------------------------------------
template<class TASK M4_PARAM(`,class F$1', `', ` ')>
class KAAPI_CLOSUREFMT(KAAPI_NUMBER_PARAMS) 
 : public DFG::WrapperClosureFormat<KAAPI_CLOSURE(KAAPI_NUMBER_PARAMS)<TASK M4_PARAM(`,F$1', `', `')> > {
public:
  KAAPI_CLOSUREFMT(KAAPI_NUMBER_PARAMS) () 
   : DFG::WrapperClosureFormat<KAAPI_CLOSURE(KAAPI_NUMBER_PARAMS)<TASK M4_PARAM(`,F$1', `', `')> >(__PRETTY_FUNCTION__, 
    &KAAPI_CLOSURE(KAAPI_NUMBER_PARAMS)<TASK M4_PARAM(`,F$1', `', `')>::srun,
    &KAAPI_CLOSURE(KAAPI_NUMBER_PARAMS)<TASK M4_PARAM(`,F$1', `', `')>::srun_onelevel,
    &KAAPI_CLOSURE(KAAPI_NUMBER_PARAMS)<TASK M4_PARAM(`,F$1', `', `')>::srun_steal,
    &KAAPI_CLOSURE(KAAPI_NUMBER_PARAMS)<TASK M4_PARAM(`,F$1', `', `')>::srun_internal,
    &KAAPI_CLOSURE(KAAPI_NUMBER_PARAMS)<TASK M4_PARAM(`,F$1', `', `')>::srun_release
  )
  {}

  int get_nparam(const RFO::Closure* c) const
  { 
    typedef KAAPI_CLOSURE(KAAPI_NUMBER_PARAMS)<TASK M4_PARAM(`,F$1',`',`')> TheClosure;
    TheClosure* clo __attribute__((unused)) = (TheClosure*)c;
    return 0
     M4_PARAM(`Trait_Type<F$1>::get_nparam(&clo->f$1)
     ', `+', `+', `');
  }

  const Util::Format* get_fmtparam(const RFO::Closure*c, int i) const
  {
    typedef KAAPI_CLOSURE(KAAPI_NUMBER_PARAMS)<TASK M4_PARAM(`,F$1',`',`')> TheClosure;
    TheClosure* clo __attribute__((unused)) = (TheClosure*)c;
    int countp __attribute__((unused)) = 0, count __attribute__((unused)) = 0;
   M4_PARAM(`count += Trait_Type<F$1>::get_nparam(&clo->f$1);
    if (i < count) return Trait_Type<F$1>::get_format(&clo->f$1, i-countp);
    countp = count; 
    ', ` ', `
    ')
    return 0;
  }

  const Util::FormatOperationCumul* get_cwfmtparam( const RFO::Closure* /*clo*/, int i) const
  {
    switch (i) {
     M4_PARAM(`case $1-1: return Trait_Type<F$1>::get_cw_format();
     ', ` ', `')
    }
    return 0;
  }

  void* get_param(const RFO::Closure* c, int i) const
  { 
    typedef KAAPI_CLOSURE(KAAPI_NUMBER_PARAMS)<TASK M4_PARAM(`,F$1',`',`')> TheClosure;
    TheClosure* clo __attribute__((unused)) = (TheClosure*)c;
    int countp __attribute__((unused)) = 0, count __attribute__((unused)) = 0;
   M4_PARAM(`count += Trait_Type<F$1>::get_nparam(&clo->f$1);
    if (i < count) return Trait_Type<F$1>::get_param(&clo->f$1, i-countp);
    countp = count; 
    ', ` ', `
    ')
    return 0;
  }

  RFO::AccessMode::Val get_mode(const RFO::Closure* c, int i) const
  { 
    typedef KAAPI_CLOSURE(KAAPI_NUMBER_PARAMS)<TASK M4_PARAM(`,F$1',`',`')> TheClosure;
    TheClosure* clo __attribute__((unused)) = (TheClosure*)c;
    int countp __attribute__((unused)) = 0, count __attribute__((unused)) = 0;
   M4_PARAM(`count += Trait_Type<F$1>::get_nparam(&clo->f$1);
    if (i < count) return Trait_Type<F$1>::get_mode(&clo->f$1, i-countp);
    countp = count; 
    ', ` ', `
    ')
    return RFO::AccessMode::A_VOID;
  }

  bool is_access(const RFO::Closure* c, int i) const
  { 
    typedef KAAPI_CLOSURE(KAAPI_NUMBER_PARAMS)<TASK M4_PARAM(`,F$1',`',`')> TheClosure;
    TheClosure* clo __attribute__((unused)) = (TheClosure*)c;
    int countp __attribute__((unused)) = 0, count __attribute__((unused)) = 0;
   M4_PARAM(`count += Trait_Type<F$1>::get_nparam(&clo->f$1);
    if (i < count) return Trait_Type<F$1>::is_access(&clo->f$1, i-countp);
    countp = count; 
    ', ` ', `
    ')
    return false;
  }

  void copy( Util::InterfaceAllocator* a, void* dest, const void* src, size_t /*count*/ ) const
  {
    typedef KAAPI_CLOSURE(KAAPI_NUMBER_PARAMS)<TASK M4_PARAM(`,F$1',`',`')> TheClosure;
    TheClosure* clo_dest = (TheClosure*)dest;
    TheClosure* clo_src = (TheClosure*)src;
    clo_dest->RFO::Closure::operator=( *clo_src );
    M4_PARAM(`Trait_Type<F$1>::copy(a, &clo_dest->f$1, &clo_src->f$1);
    ', `', `')
  }

  void write( Util::OStream& s_out, const void* val, size_t count ) const
  {
    typedef KAAPI_CLOSURE(KAAPI_NUMBER_PARAMS)<TASK M4_PARAM(`,F$1',`',`')> TheClosure;
    TheClosure* clo = (TheClosure*)val;
    for (unsigned int i=0; i<count; ++i)
    {
      M4_PARAM(`Trait_Type<F$1>::write(s_out, &clo->f$1);
      ', `', `')
      DFG::ClosureFormat::write(s_out, clo, 1);
      ++clo;
    }
  }

  void read( Util::IStream& s_in, void* val, size_t count ) const
  {
    typedef KAAPI_CLOSURE(KAAPI_NUMBER_PARAMS)<TASK M4_PARAM(`,F$1',`',`')> TheClosure;
    TheClosure* clo = (TheClosure*)val;
    for (unsigned int i=0; i<count; ++i)
    {
      M4_PARAM(`Trait_Type<F$1>::read(s_in, &clo->f$1);
      ', `', `')
      DFG::ClosureFormat::read(s_in, clo, 1);
      clo->set_run ( &TheClosure::srun );
      clo->set_format( &theformat );
      ++clo;
    }
  }

  static const KAAPI_CLOSUREFMT(KAAPI_NUMBER_PARAMS)<TASK M4_PARAM(`,F$1', `', `')> theformat;
};

template<class TASK M4_PARAM(`, class F$1', `', `')>
const KAAPI_CLOSUREFMT(KAAPI_NUMBER_PARAMS)<TASK M4_PARAM(`,F$1', `', `')>
   KAAPI_CLOSUREFMT(KAAPI_NUMBER_PARAMS)<TASK M4_PARAM(`,F$1', `', `')>::theformat;

struct KAAPI_INITFORMATCLOSURE(KAAPI_NUMBER_PARAMS) {
  template<class TASK, class ATTR M4_PARAM(`, class F$1, class E$1',`',`')>
  static KAAPI_CLOSURE(KAAPI_NUMBER_PARAMS)<TASK M4_PARAM(`,F$1',`',`')>* doit(
    Thread* thread,
    ATTR& _attr,
    void (TASK::*)( M4_PARAM(`F$1', `', `,') )
  M4_PARAM(`, const E$1& e$1
  ', `', `')
    )
  {
    typedef KAAPI_CLOSURE(KAAPI_NUMBER_PARAMS)<TASK M4_PARAM(`,F$1',`',`')> TheClosure;
    TheClosure* clo = new (thread->allocate(sizeof(TheClosure))) TheClosure;
    clo->clear();
    M4_PARAM(`Trait_Link<E$1,typename TheClosure::type_f$1>::doit( thread, clo, &e$1, &clo->f$1 );
    ', `', `')
    M4_PARAM(`Trait_Constraint<typename TheClosure::type_f$1>::local
      ', `if (', ` || ')
    M4_PARAM(`', `) { clo->set_local(); clo->unset_stealable(); } ', `')

    clo->set_format( &KAAPI_CLOSUREFMT(KAAPI_NUMBER_PARAMS)<TASK M4_PARAM(`,F$1', `', `')>::theformat );
    clo->set_run( KAAPI_CLOSURE(KAAPI_NUMBER_PARAMS)<TASK M4_PARAM(`,F$1', `', `')>::srun );
    thread->push( thread->top(), _attr(thread,clo) );
    return clo;
  }
};


template<class TASK M4_PARAM(`class F$1',`,',`,')>
inline const RFO::ClosureFormat* HandlerGetClosureFormat( 
    void (TASK::*)( M4_PARAM(`F$1', `', `,') )
  )
  {
    return &KAAPI_CLOSUREFMT(KAAPI_NUMBER_PARAMS)<TASK M4_PARAM(`,F$1', `', `')>::theformat;
  }

