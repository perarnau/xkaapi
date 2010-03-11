// --------------------------------------------------------------------
/* KAAPI_NUMBER_PARAMS is the number of possible parameters */
template<>
struct Task<KAAPI_NUMBER_PARAMS> {
  ifelse(KAAPI_NUMBER_PARAMS,0,`',`template<M4_PARAM(`class F$1', `', `, ')>')
  struct Signature { 
    M4_PARAM(`typedef typename TraitUAMParam<F$1>::uamttype_t uamttype$1_t;
    ', `', `')
    M4_PARAM(`typedef typename TraitUAMParam<F$1>::mode_t mode$1_t;
    ', `', `')
    M4_PARAM(`typedef typename uamttype$1_t::template UAMParam<TYPE_INTASK>::type_t inclosure$1_t;
    ', `', `')
    M4_PARAM(`typedef typename uamttype$1_t::template UAMParam<mode$1_t>::type_t formal$1_t;
    ', `', `')
    void operator() ( Thread* thread M4_PARAM(`, formal$1_t', `', `') ) {}
    void dummy_method_to_have_formal_param_type ( Thread* thread M4_PARAM(`, F$1 f$1', `', `') ){}
  };
};


// --------------------------------------------------------------------
/* Kaapi closure representation */
ifelse(KAAPI_NUMBER_PARAMS,0,`',`template<M4_PARAM(`typename TraitUAMType$1', `', `, ')>')
struct KAAPI_TASKARG(KAAPI_NUMBER_PARAMS) { 
 M4_PARAM(`typedef typename TraitUAMType$1::template UAMParam<TYPE_INTASK>::type_t inclosure$1_t;
  ', ` ', `')
 M4_PARAM(`inclosure$1_t f$1;
  ', ` ', `')
};


// --------------------------------------------------------------------
/* Body generators */
template<bool hasstackparam, class TASK M4_PARAM(`,typename TraitUAMParam_F$1', `', ` ')>
struct KAAPIWRAPPERCPUBODY(KAAPI_NUMBER_PARAMS) {};


/* Kaapi binder to call task with stack args */
template<class TASK M4_PARAM(`,typename TraitUAMParam_F$1', `', ` ')>
struct KAAPIWRAPPERCPUBODY(KAAPI_NUMBER_PARAMS)<true, TASK M4_PARAM(`, TraitUAMParam_F$1', `', ` ')> {
  M4_PARAM(`typedef typename TraitUAMParam_F$1::uamttype_t uamttype$1_t;
  ', `', `')
  M4_PARAM(`typedef typename TraitUAMParam_F$1::mode_t mode$1_t;
  ', `', `')
  M4_PARAM(`typedef typename uamttype$1_t::template UAMParam<mode$1_t>::type_t formal$1_t;
  ', `', `')
  typedef KAAPI_TASKARG(KAAPI_NUMBER_PARAMS) ifelse(KAAPI_NUMBER_PARAMS,0,`',`<M4_PARAM(`uamttype$1_t', `', `,')>') TaskArg_t;

  static TaskBodyCPU<TASK> dummy;
  static void body(void* taskarg, kaapi_thread_t* thread)
  {
    ifelse(KAAPI_NUMBER_PARAMS,0,`',`TaskArg_t* args = (TaskArg_t*)taskarg;')
    dummy( (Thread*)thread M4_PARAM(`, (formal$1_t)args->f$1', `', `'));
  }
};
template<class TASK M4_PARAM(`,typename TraitUAMParam_F$1', `', ` ')>
TaskBodyCPU<TASK> KAAPIWRAPPERCPUBODY(KAAPI_NUMBER_PARAMS)<true, TASK M4_PARAM(`, TraitUAMParam_F$1', `', ` ')>::dummy;


/* Kaapi binder to call task without stack args */
template<class TASK M4_PARAM(`,typename TraitUAMParam_F$1', `', ` ')>
struct KAAPIWRAPPERCPUBODY(KAAPI_NUMBER_PARAMS)<false, TASK M4_PARAM(`, TraitUAMParam_F$1', `', ` ')> {
  M4_PARAM(`typedef typename TraitUAMParam_F$1::uamttype_t uamttype$1_t;
  ', `', `')
  M4_PARAM(`typedef typename TraitUAMParam_F$1::mode_t mode$1_t;
  ', `', `')
  M4_PARAM(`typedef typename uamttype$1_t::template UAMParam<mode$1_t>::type_t formal$1_t;
  ', `', `')
  typedef KAAPI_TASKARG(KAAPI_NUMBER_PARAMS) ifelse(KAAPI_NUMBER_PARAMS,0,`',`<M4_PARAM(`uamttype$1_t', `', `,')>') TaskArg_t;

  static TaskBodyCPU<TASK> dummy;
  static void body(void* taskarg, kaapi_thread_t* thread)
  {
    ifelse(KAAPI_NUMBER_PARAMS,0,`',`TaskArg_t* args = (TaskArg_t*)taskarg;')
    dummy( M4_PARAM(`(formal$1_t)args->f$1', `', `,'));
  }
};
template<class TASK M4_PARAM(`,typename TraitUAMParam_F$1', `', ` ')>
TaskBodyCPU<TASK> KAAPIWRAPPERCPUBODY(KAAPI_NUMBER_PARAMS)<false, TASK M4_PARAM(`, TraitUAMParam_F$1', `', ` ')>::dummy;


template<bool hasstackparam, class TASK M4_PARAM(`,typename TraitUAMParam_F$1', `', ` ')>
struct KAAPIWRAPPERGPUBODY(KAAPI_NUMBER_PARAMS) {};


/* Kaapi binder to call task with stack args */
template<class TASK M4_PARAM(`,typename TraitUAMParam_F$1', `', ` ')>
struct KAAPIWRAPPERGPUBODY(KAAPI_NUMBER_PARAMS)<true, TASK M4_PARAM(`, TraitUAMParam_F$1', `', ` ')> {
  M4_PARAM(`typedef typename TraitUAMParam_F$1::uamttype_t uamttype$1_t;
  ', `', `')
  M4_PARAM(`typedef typename TraitUAMParam_F$1::mode_t mode$1_t;
  ', `', `')
  M4_PARAM(`typedef typename uamttype$1_t::template UAMParam<mode$1_t>::type_t formal$1_t;
  ', `', `')
  typedef KAAPI_TASKARG(KAAPI_NUMBER_PARAMS) ifelse(KAAPI_NUMBER_PARAMS,0,`',`<M4_PARAM(`uamttype$1_t', `', `,')>') TaskArg_t;

  /* with stack parameters */
  static TaskBodyGPU<TASK> dummy;
  static void body(void* taskarg, kaapi_thread_t* thread)
  {
    ifelse(KAAPI_NUMBER_PARAMS,0,`',`TaskArg_t* args = (TaskArg_t*)taskarg;')
    dummy( (Thread*)thread M4_PARAM(`, (formal$1_t)args->f$1', `', `'));
  }
};
template<class TASK M4_PARAM(`,typename TraitUAMParam_F$1', `', ` ')>
TaskBodyGPU<TASK>  KAAPIWRAPPERGPUBODY(KAAPI_NUMBER_PARAMS)<true, TASK M4_PARAM(`, TraitUAMParam_F$1', `', ` ')>::dummy;


/* Kaapi binder to call task without stack args */
template<class TASK M4_PARAM(`,typename TraitUAMParam_F$1', `', ` ')>
struct KAAPIWRAPPERGPUBODY(KAAPI_NUMBER_PARAMS)<false, TASK M4_PARAM(`, TraitUAMParam_F$1', `', ` ')> {
  M4_PARAM(`typedef typename TraitUAMParam_F$1::uamttype_t uamttype$1_t;
  ', `', `')
  M4_PARAM(`typedef typename TraitUAMParam_F$1::mode_t mode$1_t;
  ', `', `')
  M4_PARAM(`typedef typename uamttype$1_t::template UAMParam<mode$1_t>::type_t formal$1_t;
  ', `', `')
  typedef KAAPI_TASKARG(KAAPI_NUMBER_PARAMS) ifelse(KAAPI_NUMBER_PARAMS,0,`',`<M4_PARAM(`uamttype$1_t', `', `,')>') TaskArg_t;

  /* */
  static TaskBodyGPU<TASK> dummy;
  static void body(void* taskarg, kaapi_thread_t* thread)
  {
    ifelse(KAAPI_NUMBER_PARAMS,0,`',`TaskArg_t* args = (TaskArg_t*)taskarg;')
    dummy( M4_PARAM(`(formal$1_t)args->f$1', `', `,'));
  }
};
template<class TASK M4_PARAM(`,typename TraitUAMParam_F$1', `', ` ')>
TaskBodyGPU<TASK>  KAAPIWRAPPERGPUBODY(KAAPI_NUMBER_PARAMS)<false, TASK M4_PARAM(`, TraitUAMParam_F$1', `', ` ')>::dummy;



template<class TASK M4_PARAM(`,typename TraitUAMParam_F$1', `', ` ')>
struct KAAPI_FORMATCLOSURE(KAAPI_NUMBER_PARAMS) {

  M4_PARAM(`typedef typename TraitUAMParam_F$1::uamttype_t uamttype$1_t;
  ', `', `')
  M4_PARAM(`typedef typename uamttype$1_t::template UAMParam<TYPE_INTASK>::type_t inclosure$1_t;
  ', `', `')
  M4_PARAM(`typedef typename uamttype$1_t::typeformat_t typeformat$1_t;
  ', `', `')
  typedef KAAPI_TASKARG(KAAPI_NUMBER_PARAMS) ifelse(KAAPI_NUMBER_PARAMS,0,`',`<M4_PARAM(`uamttype$1_t', `', `,')>') TaskArg_t;

  static kaapi_format_t    format;
  static kaapi_format_id_t fmid;
  static kaapi_format_t*   getformat() { return &format; }
  static kaapi_task_bodyid_t default_body;

  static kaapi_format_t* registerformat()
  {
    /* here we assume no concurrency during startup calls of the library that initialize format objects */
    if (fmid != 0) return &format;
    default_body = 0;
    
    ifelse(KAAPI_NUMBER_PARAMS,0,`',`static kaapi_access_mode_t   array_mode[KAAPI_NUMBER_PARAMS];')
    ifelse(KAAPI_NUMBER_PARAMS,0,`',`static kaapi_offset_t        array_offset[KAAPI_NUMBER_PARAMS];')
    ifelse(KAAPI_NUMBER_PARAMS,0,`',`static const kaapi_format_t* array_format[KAAPI_NUMBER_PARAMS];')
    TaskArg_t* dummy;
    M4_PARAM(`array_mode[$1-1] = (kaapi_access_mode_t)TraitUAMParam_F$1::mode_t::value;
    ',`', `')
    M4_PARAM(`array_offset[$1-1] = (char*)&dummy->f$1 - (char*)dummy; /* BUG ? offsetof(TaskArg_t, f$1); */
    ',`', `')
    M4_PARAM(`array_format[$1-1] = WrapperFormat<typeformat$1_t>::get_format();
    ',`', `')
    static std::string task_name = std::string("__Z")+std::string(typeid(TASK).name());
    fmid = kaapi_format_taskregister( 
          &getformat, 
          0,
          0, 
          task_name.c_str(),
          sizeof(TaskArg_t),
          KAAPI_NUMBER_PARAMS,
          ifelse(KAAPI_NUMBER_PARAMS,0,`0',`array_mode'),
          ifelse(KAAPI_NUMBER_PARAMS,0,`0',`array_offset'),
          ifelse(KAAPI_NUMBER_PARAMS,0,`0',`array_format')
      );
    return &format;
  }

};


template<class TASK M4_PARAM(`,typename TraitUAMParam_F$1', `', ` ')>
struct KAAPI_INITFORMATCLOSURE(KAAPI_NUMBER_PARAMS) {

  M4_PARAM(`typedef typename TraitUAMParam_F$1::uamttype_t uamttype$1_t;
  ', `', `')
  M4_PARAM(`typedef typename TraitUAMParam_F$1::mode_t mode$1_t;
  ', `', `')
  M4_PARAM(`typedef typename uamttype$1_t::template UAMParam<TYPE_INTASK>::type_t inclosure$1_t;
  ', `', `')
  M4_PARAM(`typedef typename uamttype$1_t::template UAMParam<mode$1_t>::type_t formal$1_t;
  ', `', `')

  static kaapi_task_body_t registercpubody( kaapi_format_t* fmt, void (TaskBodyCPU<TASK>::*method)( M4_PARAM(`formal$1_t', `', `,') ) )
  {
    typedef void (TASK::*type_default_t)(Thread* M4_PARAM(`, formal$1_t', `', `'));
    type_default_t f_default = &TASK::operator();
    if ((type_default_t)method == f_default) return 0;
    return &KAAPIWRAPPERCPUBODY(KAAPI_NUMBER_PARAMS)<false, TASK M4_PARAM(`,TraitUAMParam_F$1', `', ` ')>::body;
  }

  static kaapi_task_body_t registercpubody( kaapi_format_t* fmt, void (TaskBodyCPU<TASK>::*method)( Thread* M4_PARAM(`, formal$1_t', `', `') ) )
  {
    typedef void (TASK::*type_default_t)(Thread* M4_PARAM(`, formal$1_t', `', `'));
    type_default_t f_default = &TASK::operator();
    if ((type_default_t)method == f_default) return 0;
    return &KAAPIWRAPPERCPUBODY(KAAPI_NUMBER_PARAMS)<true, TASK M4_PARAM(`,TraitUAMParam_F$1', `', ` ')>::body;
  }

  static kaapi_task_body_t registergpubody( kaapi_format_t* fmt, void (TaskBodyGPU<TASK>::*method)( M4_PARAM(`formal$1_t', `', `,') ) )
  {
    typedef void (TASK::*type_default_t)(Thread* M4_PARAM(`, formal$1_t', `', `'));
    type_default_t f_default = &TASK::operator();
    if ((type_default_t)method == f_default) return 0;
    return &KAAPIWRAPPERGPUBODY(KAAPI_NUMBER_PARAMS)<false, TASK M4_PARAM(`,TraitUAMParam_F$1', `', ` ')>::body;
  }

  static kaapi_task_body_t registergpubody( kaapi_format_t* fmt, void (TaskBodyGPU<TASK>::*method)( Thread* M4_PARAM(`, formal$1_t', `', `') ) )
  {
    typedef void (TASK::*type_default_t)(Thread* M4_PARAM(`, formal$1_t', `', `'));
    type_default_t f_default = &TASK::operator();
    if ((type_default_t)method == f_default) return 0;
    return &KAAPIWRAPPERGPUBODY(KAAPI_NUMBER_PARAMS)<true, TASK M4_PARAM(`,TraitUAMParam_F$1', `', ` ')>::body;
  }

  static kaapi_format_t* registerbodycpu( kaapi_format_t* fmt, void (TASK::*method)( Thread* thread M4_PARAM(`, formal$1_t', `', `') ) )
  {
    kaapi_format_taskregister_body( fmt, registercpubody( fmt, &TaskBodyCPU<TASK>::operator() ), KAAPI_PROC_TYPE_CPU );
    /* here we assume CPU is the running processor */
    KAAPI_FORMATCLOSURE(KAAPI_NUMBER_PARAMS)<TASK M4_PARAM(`, TraitUAMParam_F$1', `', `')>::default_body = 
      fmt->default_body = fmt->entrypoint[KAAPI_PROC_TYPE_CPU];
    return fmt;
  }
  static kaapi_format_t* registerbodygpu( kaapi_format_t* fmt, void (TASK::*method)( Thread* thread M4_PARAM(`, formal$1_t', `', `') ) )
  {
    kaapi_format_taskregister_body( fmt, registergpubody( fmt, &TaskBodyGPU<TASK>::operator() ), KAAPI_PROC_TYPE_GPU );
    return fmt;
  }
  static kaapi_format_t* registerbodies( kaapi_format_t* fmt, void (TASK::*method)( Thread* thread M4_PARAM(`, formal$1_t', `', `') ) )
  {
    registerbodycpu( fmt, method );
    registerbodygpu( fmt, method );
    return fmt;
  }
};


template<class TASK M4_PARAM(`,typename TraitUAMParam_F$1', `', ` ')>
kaapi_format_t KAAPI_FORMATCLOSURE(KAAPI_NUMBER_PARAMS)<TASK M4_PARAM(`,TraitUAMParam_F$1', `', ` ')>::format;

template<class TASK M4_PARAM(`,typename TraitUAMParam_F$1', `', ` ')>
kaapi_format_id_t KAAPI_FORMATCLOSURE(KAAPI_NUMBER_PARAMS)<TASK M4_PARAM(`,TraitUAMParam_F$1', `', ` ')>::fmid = 0;


template<class TASK M4_PARAM(`,typename TraitUAMParam_F$1', `', ` ')>
kaapi_task_bodyid_t KAAPI_FORMATCLOSURE(KAAPI_NUMBER_PARAMS)<TASK M4_PARAM(`,TraitUAMParam_F$1', `', ` ')>::default_body =
    KAAPI_INITFORMATCLOSURE(KAAPI_NUMBER_PARAMS)<TASK M4_PARAM(`,TraitUAMParam_F$1', `', ` ')>::registerbodies(
          KAAPI_FORMATCLOSURE(KAAPI_NUMBER_PARAMS)<TASK M4_PARAM(`,TraitUAMParam_F$1', `', ` ')>::registerformat(), 
          &TASK::operator()
    )->default_body;


  template<class TASK, class SIGNATURE  M4_PARAM(`, class F$1', `', `')>
  void DoRegisterBodyCPU( void (SIGNATURE::*)( Thread* M4_PARAM(`, F$1', `', `') ) )
  {
      KAAPI_INITFORMATCLOSURE(KAAPI_NUMBER_PARAMS)<TASK M4_PARAM(`,TraitUAMParam<F$1> ', `', ` ')>::registerbodycpu(
          KAAPI_FORMATCLOSURE(KAAPI_NUMBER_PARAMS)<TASK M4_PARAM(`,TraitUAMParam<F$1> ', `', ` ')>::registerformat(), 
          &TASK::operator()
      );
  }
  template<class TASK, class SIGNATURE  M4_PARAM(`, class F$1', `', `')>
  void DoRegisterBodyGPU( void (SIGNATURE::*)( Thread* M4_PARAM(`, F$1', `', `') ) )
  {
      KAAPI_INITFORMATCLOSURE(KAAPI_NUMBER_PARAMS)<TASK M4_PARAM(`,TraitUAMParam<F$1> ', `', ` ')>::registerbodygpu(
          KAAPI_FORMATCLOSURE(KAAPI_NUMBER_PARAMS)<TASK M4_PARAM(`,TraitUAMParam<F$1> ', `', ` ')>::registerformat(), 
          &TASK::operator()
      );
  }
