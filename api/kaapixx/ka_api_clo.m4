// --------------------------------------------------------------------
// Kaapi closure representation
ifelse(KAAPI_NUMBER_PARAMS,0,`',`template<M4_PARAM(`typename TraitFormal$1', `', `, ')>')
struct KAAPI_TASKARG(KAAPI_NUMBER_PARAMS) { 
 M4_PARAM(`typedef typename TraitFormal$1::type_inclosure_t inclosure$1_t;
  ', ` ', `')
 static const bool is_static = true 
      M4_PARAM(`&& TraitFormal$1::is_static
      ', ` ', `')
      ;
 M4_PARAM(`inclosure$1_t f$1;
  ', ` ', `')
};


// --------------------------------------------------------------------
// KAAPI_NUMBER_PARAMS is the number of possible parameters
template<>
struct Task<KAAPI_NUMBER_PARAMS> {
  public:
  ifelse(KAAPI_NUMBER_PARAMS,0,`',`template<M4_PARAM(`class F$1', `', `, ')>')
  struct Signature { 
    M4_PARAM(`typedef typename TraitFormalParam<F$1>::type_inclosure_t inclosure$1_t;
    ', `', `')
    M4_PARAM(`typedef typename TraitFormalParam<F$1>::mode_t mode$1_t;
    ', `', `')
    M4_PARAM(`typedef F$1 signature$1_t;
    ', `', `')
    typedef KAAPI_TASKARG(KAAPI_NUMBER_PARAMS) ifelse(KAAPI_NUMBER_PARAMS,0,`',`<M4_PARAM(`TraitFormalParam<F$1>', `', `,') >') TaskArg_t;
    
    void operator() ( THIS_TYPE_IS_USED_ONLY_INTERNALLY dummy M4_PARAM(`, signature$1_t', `', `') ) {}
    void dummy_method_to_have_formal_param_type ( Thread* thread M4_PARAM(`, signature$1_t f$1', `', `') ){}
    static const int nargs = KAAPI_NUMBER_PARAMS;
  };
};

// --------------------------------------------------------------------
// KAAPI_NUMBER_PARAMS is the number of possible parameters. Same
template<>
struct Splitter<KAAPI_NUMBER_PARAMS> {
  void operator()( THIS_TYPE_IS_USED_ONLY_INTERNALLY ) {}
};

template<class TASK M4_PARAM(`,typename F$1', `', ` ')>
kaapi_yes_type kaapi_is_func_splitter( int (TASK::*)(StealContext*, int, ListRequest::iterator, ListRequest::iterator M4_PARAM(`, F$1', `', ` ')) );


// --------------------------------------------------------------------
// KAAPI_NUMBER_PARAMS is the number of possible parameters
template<class TASK M4_PARAM(`,typename TraitFormalParam$1', `', ` ')>
struct KAAPIWRAPPER_SPLITTER(KAAPI_NUMBER_PARAMS) { 
  typedef KAAPI_TASKARG(KAAPI_NUMBER_PARAMS) ifelse(KAAPI_NUMBER_PARAMS,0,`',`<M4_PARAM(`TraitFormalParam$1', `', `,')>') TaskArg_t;
  M4_PARAM(`typedef typename TraitFormalParam$1::formal_t formal$1_t;
  ', ` ', `')
  static TaskSplitter<TASK> dummy;
  static int splitter( 
      struct kaapi_task_t* victim_task,
      void* task_arg,
      struct kaapi_listrequest_t* lr,
      struct kaapi_listrequest_iterator_t* lri
  )
  {
    int nreq = kaapi_api_listrequest_iterator_count(lri);
    ifelse(KAAPI_NUMBER_PARAMS,0,`',`TaskArg_t* args = (TaskArg_t*)task_arg;')
    dummy( 
        (StealContext*)victim_task,
        nreq,
        ListRequest::iterator(lr,lri), ListRequest::iterator(lr,0)
        M4_PARAM(`(formal$1_t)args->f$1', `,', `,')
    );
    return 0;
  };
};

template<class TASK M4_PARAM(`,typename TraitFormalParam$1', `', ` ')>
TaskSplitter<TASK> KAAPIWRAPPER_SPLITTER(KAAPI_NUMBER_PARAMS)<TASK M4_PARAM(`,TraitFormalParam$1', `', `')>::dummy;

// --------------------------------------------------------------------
// KAAPI_NUMBER_PARAMS is the number of possible parameters
template<bool flag, class ATTR, class TASK M4_PARAM(`,typename F$1', `', ` ')>
struct KAAPI_PUSHER(KAAPI_NUMBER_PARAMS) {};

template<class ATTR, class TASK M4_PARAM(`,typename TraitFormalParam$1', `', ` ')>
struct KAAPI_PUSHER(KAAPI_NUMBER_PARAMS)<true, ATTR, TASK M4_PARAM(`,TraitFormalParam$1', `', ` ')> 
{ /* forget attr: not used if adaptive task is pushed */
  static void push( const ATTR& attr, kaapi_thread_t* thread )
  {
    kaapi_thread_pushtask_adaptive( thread, 
      KAAPIWRAPPER_SPLITTER(KAAPI_NUMBER_PARAMS)<TASK M4_PARAM(`,TraitFormalParam$1', `', ` ')>::splitter
    );
  }
};

template<class ATTR, class TASK M4_PARAM(`,typename TraitFormalParam$1', `', ` ')>
struct KAAPI_PUSHER(KAAPI_NUMBER_PARAMS)<false, ATTR, TASK M4_PARAM(`,TraitFormalParam$1', `', ` ')> 
{
  static void push( const ATTR& attr, kaapi_thread_t* thread )
  {
    attr( thread );
  }
};

template<bool flag, class TASK M4_PARAM(`,typename F$1', `', ` ')>
struct KAAPI_REQPUSHER(KAAPI_NUMBER_PARAMS) {};

template<class TASK M4_PARAM(`,typename TraitFormalParam$1', `', ` ')>
struct KAAPI_REQPUSHER(KAAPI_NUMBER_PARAMS)<true, TASK M4_PARAM(`,TraitFormalParam$1', `', ` ')> 
{ /* forget attr: not used if adaptive task is pushed */
  static void push( kaapi_request_t* req, kaapi_task_t* victim_task, int flag )
  {
    kaapi_request_pushtask_adaptive( req, victim_task, 
      KAAPIWRAPPER_SPLITTER(KAAPI_NUMBER_PARAMS)<TASK M4_PARAM(`TraitFormalParam$1', `,', `,') >::splitter, flag 
    );
  }
};

template<class TASK M4_PARAM(`,typename TraitFormalParam$1', `', ` ')>
struct KAAPI_REQPUSHER(KAAPI_NUMBER_PARAMS)<false, TASK M4_PARAM(`,TraitFormalParam$1', `', ` ')> 
{
  static void push( kaapi_request_t* req, kaapi_task_t* victim_task, int flag )
  {
    kaapi_request_pushtask( req, victim_task );
  }
};


// --------------------------------------------------------------------
// Body generators: 1 -> only user args, 2 -> Thread*, 3-> steal context, 4-> sched info
template<int type, class TASK M4_PARAM(`,typename TraitFormalParam$1', `', ` ')>
struct KAAPIWRAPPER_CPU_BODY(KAAPI_NUMBER_PARAMS) {};


// Kaapi binder to call task without stack args
template<class TASK M4_PARAM(`,typename TraitFormalParam$1', `', ` ')>
struct KAAPIWRAPPER_CPU_BODY(KAAPI_NUMBER_PARAMS)<1, TASK M4_PARAM(`, TraitFormalParam$1', `', ` ')> {
  typedef KAAPI_TASKARG(KAAPI_NUMBER_PARAMS) ifelse(KAAPI_NUMBER_PARAMS,0,`',`<M4_PARAM(`TraitFormalParam$1', `', `,')>') TaskArg_t;
  M4_PARAM(`typedef typename TraitFormalParam$1::formal_t formal$1_t;
  ', ` ', `')
  static TaskBodyCPU<TASK> dummy;
  static void body(void* taskarg, kaapi_thread_t* thread)
  {
    ifelse(KAAPI_NUMBER_PARAMS,0,`',`TaskArg_t* args = (TaskArg_t*)taskarg;')
    dummy( M4_PARAM(`(formal$1_t)args->f$1', `', `,'));
//    dummy( M4_PARAM(`args->f$1', `', `,'));
  }
  static void bodywithhandle(void* taskarg, kaapi_thread_t* thread)
  {
    ifelse(KAAPI_NUMBER_PARAMS,0,`',`TaskArg_t* args = (TaskArg_t*)taskarg;')
    dummy( M4_PARAM(`TraitFormalParam$1::handle2data(&args->f$1)', `', `,'));
  }
};
template<class TASK M4_PARAM(`,typename TraitFormalParam$1', `', ` ')>
TaskBodyCPU<TASK> KAAPIWRAPPER_CPU_BODY(KAAPI_NUMBER_PARAMS)<1, TASK M4_PARAM(`, TraitFormalParam$1', `', ` ')>::dummy;

// Kaapi binder to call task with thread arg
template<class TASK M4_PARAM(`,typename TraitFormalParam$1', `', ` ')>
struct KAAPIWRAPPER_CPU_BODY(KAAPI_NUMBER_PARAMS)<2, TASK M4_PARAM(`, TraitFormalParam$1', `', ` ')> {
  typedef KAAPI_TASKARG(KAAPI_NUMBER_PARAMS) ifelse(KAAPI_NUMBER_PARAMS,0,`',`<M4_PARAM(`TraitFormalParam$1', `', `,')>') TaskArg_t;
  M4_PARAM(`typedef typename TraitFormalParam$1::formal_t formal$1_t;
  ', ` ', `')
  static TaskBodyCPU<TASK> dummy;
  static void body(void* taskarg, kaapi_thread_t* thread)
  {
    ifelse(KAAPI_NUMBER_PARAMS,0,`',`TaskArg_t* args = (TaskArg_t*)taskarg;')
    dummy( (Thread*)thread M4_PARAM(`, (formal$1_t)args->f$1', `', `'));
  }
  static void bodywithhandle(void* taskarg, kaapi_thread_t* thread)
  {
    ifelse(KAAPI_NUMBER_PARAMS,0,`',`TaskArg_t* args = (TaskArg_t*)taskarg;')
    dummy( (Thread*)thread M4_PARAM(`,TraitFormalParam$1::handle2data(&args->f$1)', `', `'));
  }
};
template<class TASK M4_PARAM(`,typename TraitFormalParam$1', `', ` ')>
TaskBodyCPU<TASK> KAAPIWRAPPER_CPU_BODY(KAAPI_NUMBER_PARAMS)<2, TASK M4_PARAM(`, TraitFormalParam$1', `', ` ')>::dummy;

// Kaapi binder to call task with steal context args
template<class TASK M4_PARAM(`,typename TraitFormalParam$1', `', ` ')>
struct KAAPIWRAPPER_CPU_BODY(KAAPI_NUMBER_PARAMS)<3, TASK M4_PARAM(`, TraitFormalParam$1', `', ` ')> {
  typedef KAAPI_TASKARG(KAAPI_NUMBER_PARAMS) ifelse(KAAPI_NUMBER_PARAMS,0,`',`<M4_PARAM(`TraitFormalParam$1', `', `,')>') TaskArg_t;
  M4_PARAM(`typedef typename TraitFormalParam$1::formal_t formal$1_t;
  ', ` ', `')
  static TaskBodyCPU<TASK> dummy;
  static void body(void* taskarg, kaapi_thread_t* thread, kaapi_task_t* task)
  {
    ifelse(KAAPI_NUMBER_PARAMS,0,`',`TaskArg_t* args = (TaskArg_t*)taskarg;')
    dummy( (StealContext*)task M4_PARAM(`, (formal$1_t)args->f$1', `', `'));
  }
  static void bodywithhandle(void* taskarg, kaapi_thread_t* thread, kaapi_task_t* task)
  {
    ifelse(KAAPI_NUMBER_PARAMS,0,`',`TaskArg_t* args = (TaskArg_t*)taskarg;')
    dummy( (StealContext*)task M4_PARAM(`,TraitFormalParam$1::handle2data(&args->f$1)', `', `'));
  }
};
template<class TASK M4_PARAM(`,typename TraitFormalParam$1', `', ` ')>
TaskBodyCPU<TASK> KAAPIWRAPPER_CPU_BODY(KAAPI_NUMBER_PARAMS)<3, TASK M4_PARAM(`, TraitFormalParam$1', `', ` ')>::dummy;


// Kaapi binder to call task with static sched info args
template<class TASK M4_PARAM(`,typename TraitFormalParam$1', `', ` ')>
struct KAAPIWRAPPER_CPU_BODY(KAAPI_NUMBER_PARAMS)<4, TASK M4_PARAM(`, TraitFormalParam$1', `', ` ')> {
  typedef KAAPI_TASKARG(KAAPI_NUMBER_PARAMS) ifelse(KAAPI_NUMBER_PARAMS,0,`',`<M4_PARAM(`TraitFormalParam$1', `', `,')>') TaskArg_t;
  M4_PARAM(`typedef typename TraitFormalParam$1::formal_t formal$1_t;
  ', ` ', `')
  static TaskBodyCPU<TASK> dummy;
  static void body(void* taskarg, kaapi_thread_t* thread, kaapi_task_t* task)
  {
    /* task is kaapi_staticschedtask_body envelop task */
    kaapi_staticschedtask_arg_t* arg = (kaapi_staticschedtask_arg_t*)task->sp;
    kaapi_staticschedinfo_t* ssi = &arg->schedinfo;
    ifelse(KAAPI_NUMBER_PARAMS,0,`',`TaskArg_t* args = (TaskArg_t*)taskarg;')
    dummy( (StaticSchedInfo*)ssi M4_PARAM(`, (formal$1_t)args->f$1', `', `'));
  }
  static void bodywithhandle(void* taskarg, kaapi_thread_t* thread, kaapi_task_t* task)
  {
    /* task is kaapi_staticschedtask_body envelop task */
    kaapi_staticschedtask_arg_t* arg = (kaapi_staticschedtask_arg_t*)task->sp;
    kaapi_staticschedinfo_t* ssi = &arg->schedinfo;
    ifelse(KAAPI_NUMBER_PARAMS,0,`',`TaskArg_t* args = (TaskArg_t*)taskarg;')
    dummy( (StaticSchedInfo*)ssi M4_PARAM(`,TraitFormalParam$1::handle2data(&args->f$1)', `', `'));
  }
};
template<class TASK M4_PARAM(`,typename TraitFormalParam$1', `', ` ')>
TaskBodyCPU<TASK> KAAPIWRAPPER_CPU_BODY(KAAPI_NUMBER_PARAMS)<4, TASK M4_PARAM(`, TraitFormalParam$1', `', ` ')>::dummy;


#if 1
template<int type, class TASK M4_PARAM(`,typename TraitFormalParam$1', `', ` ')>
struct KAAPIWRAPPER_ALPHA_BODY(KAAPI_NUMBER_PARAMS) {};

// Kaapi binder to call task with stack args
template<class TASK M4_PARAM(`,typename TraitFormalParam$1', `', ` ')>
struct KAAPIWRAPPER_ALPHA_BODY(KAAPI_NUMBER_PARAMS)<2, TASK M4_PARAM(`, TraitFormalParam$1', `', ` ')> {
  typedef KAAPI_TASKARG(KAAPI_NUMBER_PARAMS) ifelse(KAAPI_NUMBER_PARAMS,0,`',`<M4_PARAM(`TraitFormalParam$1', `', `,')>') TaskArg_t;
  M4_PARAM(`typedef typename TraitFormalParam$1::formal_t formal$1_t;
  ', ` ', `')
  // with stack parameters
  static TaskBodyAlpha<TASK> dummy;
  static void body(void* taskarg, ka::pointer_w<float> alpha)
  {
    ifelse(KAAPI_NUMBER_PARAMS,0,`',`TaskArg_t* args = (TaskArg_t*)taskarg;')
    dummy( alpha M4_PARAM(`, (formal$1_t)args->f$1', `', `'));
  }
  static void bodywithhandle(void* taskarg, ka::pointer_w<float> alpha)
  {
    ifelse(KAAPI_NUMBER_PARAMS,0,`',`TaskArg_t* args = (TaskArg_t*)taskarg;')
    dummy( alpha M4_PARAM(`, TraitFormalParam$1::handle2data(&args->f$1)', `', `'));
  }
};
template<class TASK M4_PARAM(`,typename TraitFormalParam$1', `', ` ')>
TaskBodyAlpha<TASK>  KAAPIWRAPPER_ALPHA_BODY(KAAPI_NUMBER_PARAMS)<2, TASK M4_PARAM(`, TraitFormalParam$1', `', ` ')>::dummy;
#endif

template<int type, class TASK M4_PARAM(`,typename TraitFormalParam$1', `', ` ')>
struct KAAPIWRAPPER_GPU_BODY(KAAPI_NUMBER_PARAMS) {};


// Kaapi binder to call task without stack args
template<class TASK M4_PARAM(`,typename TraitFormalParam$1', `', ` ')>
struct KAAPIWRAPPER_GPU_BODY(KAAPI_NUMBER_PARAMS)<1, TASK M4_PARAM(`, TraitFormalParam$1', `', ` ')> {
  typedef KAAPI_TASKARG(KAAPI_NUMBER_PARAMS) ifelse(KAAPI_NUMBER_PARAMS,0,`',`<M4_PARAM(`TraitFormalParam$1', `', `,')>') TaskArg_t;
  M4_PARAM(`typedef typename TraitFormalParam$1::formal_t formal$1_t;
  ', ` ', `')
  //
  static TaskBodyGPU<TASK> dummy;
  static void body(void* taskarg, kaapi_gpustream_t gpustream)
  {
    ifelse(KAAPI_NUMBER_PARAMS,0,`',`TaskArg_t* args = (TaskArg_t*)taskarg;')
    dummy( M4_PARAM(`(formal$1_t)args->f$1', `', `,'));
  }
  static void bodywithhandle(void* taskarg, kaapi_gpustream_t gpustream)
  {
    ifelse(KAAPI_NUMBER_PARAMS,0,`',`TaskArg_t* args = (TaskArg_t*)taskarg;')
    dummy( M4_PARAM(`TraitFormalParam$1::handle2data(&args->f$1)', `', `,'));
  }
};
template<class TASK M4_PARAM(`,typename TraitFormalParam$1', `', ` ')>
TaskBodyGPU<TASK>  KAAPIWRAPPER_GPU_BODY(KAAPI_NUMBER_PARAMS)<1, TASK M4_PARAM(`, TraitFormalParam$1', `', ` ')>::dummy;



// Kaapi binder to call task with stack args
template<class TASK M4_PARAM(`,typename TraitFormalParam$1', `', ` ')>
struct KAAPIWRAPPER_GPU_BODY(KAAPI_NUMBER_PARAMS)<2, TASK M4_PARAM(`, TraitFormalParam$1', `', ` ')> {
  typedef KAAPI_TASKARG(KAAPI_NUMBER_PARAMS) ifelse(KAAPI_NUMBER_PARAMS,0,`',`<M4_PARAM(`TraitFormalParam$1', `', `,')>') TaskArg_t;
  M4_PARAM(`typedef typename TraitFormalParam$1::formal_t formal$1_t;
  ', ` ', `')
  // with stack parameters
  static TaskBodyGPU<TASK> dummy;
  static void body(void* taskarg, kaapi_gpustream_t gpustream)
  {
    ifelse(KAAPI_NUMBER_PARAMS,0,`',`TaskArg_t* args = (TaskArg_t*)taskarg;')
    dummy( (gpuStream)gpustream M4_PARAM(`, (formal$1_t)args->f$1', `', `'));
  }
  static void bodywithhandle(void* taskarg, kaapi_gpustream_t gpustream)
  {
    ifelse(KAAPI_NUMBER_PARAMS,0,`',`TaskArg_t* args = (TaskArg_t*)taskarg;')
    dummy( (gpuStream)gpustream M4_PARAM(`, TraitFormalParam$1::handle2data(&args->f$1)', `', `'));
  }
};
template<class TASK M4_PARAM(`,typename TraitFormalParam$1', `', ` ')>
TaskBodyGPU<TASK>  KAAPIWRAPPER_GPU_BODY(KAAPI_NUMBER_PARAMS)<2, TASK M4_PARAM(`, TraitFormalParam$1', `', ` ')>::dummy;

template<class TASK M4_PARAM(`,typename TraitFormalParam$1', `', ` '), bool istatic>
struct KAAPI_FORMATCLOSURE_SD(KAAPI_NUMBER_PARAMS) {};

// specialisation for static format 
template<class TASK M4_PARAM(`,typename TraitFormalParam$1', `', ` ')>
struct KAAPI_FORMATCLOSURE_SD(KAAPI_NUMBER_PARAMS)<TASK M4_PARAM(`,TraitFormalParam$1', `', ` '),true> 
{
  M4_PARAM(`typedef typename TraitFormalParam$1::type_inclosure_t inclosure$1_t;
  ', `', `')
  typedef KAAPI_TASKARG(KAAPI_NUMBER_PARAMS) ifelse(KAAPI_NUMBER_PARAMS,0,`',`<M4_PARAM(`TraitFormalParam$1', `', `,')>') TaskArg_t;

  static kaapi_format_t* registerformat()
  {
    // select at compile time between static format or dynamic format if one of the parameters
    // has variable size (array etc...
    
    // here we assume no concurrency during startup calls of the library that initialize format objects
    ifelse(KAAPI_NUMBER_PARAMS,0,`',`static kaapi_access_mode_t   array_mode[KAAPI_NUMBER_PARAMS];')
    ifelse(KAAPI_NUMBER_PARAMS,0,`',`static kaapi_offset_t        array_offset_data[KAAPI_NUMBER_PARAMS];')
    ifelse(KAAPI_NUMBER_PARAMS,0,`',`static kaapi_offset_t        array_offset_version[KAAPI_NUMBER_PARAMS];')
    ifelse(KAAPI_NUMBER_PARAMS,0,`',`static const kaapi_format_t* array_format[KAAPI_NUMBER_PARAMS];')
    ifelse(KAAPI_NUMBER_PARAMS,0,`',`static kaapi_memory_view_t   array_view[KAAPI_NUMBER_PARAMS];')
    ifelse(KAAPI_NUMBER_PARAMS,0,`',`static kaapi_reducor_t       array_reducor[KAAPI_NUMBER_PARAMS];')
    ifelse(KAAPI_NUMBER_PARAMS,0,`',`static kaapi_redinit_t       array_redinit[KAAPI_NUMBER_PARAMS];')
    TaskArg_t* dummy =0;
    M4_PARAM(`array_mode[$1-1] = (kaapi_access_mode_t)TraitFormalParam$1::mode_t::value;
    ',`', `')
    M4_PARAM(`array_offset_data[$1-1] = (char*)TraitFormalParam$1::get_data( &dummy->f$1, 0 ) - (char*)dummy; 
    ',`', `')
    M4_PARAM(`array_offset_version[$1-1] = (char*)TraitFormalParam$1::get_version( &dummy->f$1, 0 ) - (char*)dummy;
    ',`', `')
    M4_PARAM(`array_format[$1-1] = WrapperFormat<typename TraitFormalParam$1::type_t>::get_c_format();
    ',`', `')
    M4_PARAM(`array_view[$1-1]   = TraitFormalParam$1::get_view_param(&dummy->f$1, $1-1);
    ',`', `')
    M4_PARAM(`array_reducor[$1-1]= &TraitFormalParam$1::reducor_fnc;
    ',`', `')
    M4_PARAM(`array_redinit[$1-1]= &TraitFormalParam$1::redinit_fnc;
    ',`', `')
    static FormatTask task_fmt( 
          typeid(TASK).name(),
          sizeof(TaskArg_t),
          KAAPI_NUMBER_PARAMS,
          ifelse(KAAPI_NUMBER_PARAMS,0,`0',`array_mode'),
          ifelse(KAAPI_NUMBER_PARAMS,0,`0',`array_offset_data'),
          ifelse(KAAPI_NUMBER_PARAMS,0,`0',`array_offset_version'),
          ifelse(KAAPI_NUMBER_PARAMS,0,`0',`array_format'),
          ifelse(KAAPI_NUMBER_PARAMS,0,`0',`array_view'),
          ifelse(KAAPI_NUMBER_PARAMS,0,`0',`array_reducor'),
          ifelse(KAAPI_NUMBER_PARAMS,0,`0',`array_redinit'),
          0
    );
      
    return task_fmt.get_c_format();
  }
};


// specialisation for dynamic format 
template<class TASK M4_PARAM(`,typename TraitFormalParam$1', `', ` ')>
struct KAAPI_FORMATCLOSURE_SD(KAAPI_NUMBER_PARAMS)<TASK  M4_PARAM(`,TraitFormalParam$1', `', ` '),false> 
{
  M4_PARAM(`typedef typename TraitFormalParam$1::type_inclosure_t inclosure$1_t;
  ', `', `')
  typedef KAAPI_TASKARG(KAAPI_NUMBER_PARAMS) ifelse(KAAPI_NUMBER_PARAMS,0,`',`<M4_PARAM(`TraitFormalParam$1', `', `,')>') TaskArg_t;
  

  static size_t get_count_params(const struct kaapi_format_t*, const void* _taskarg)
  {
    const TaskArg_t* taskarg = static_cast<const TaskArg_t*>(_taskarg);
    size_t count __attribute__((unused)) = 0;
   M4_PARAM(`count += TraitFormalParam$1::get_nparam(&taskarg->f$1);
    ', ` ', `
    ')
    return count;
  }
  
  static kaapi_access_mode_t get_mode_param(const struct kaapi_format_t* fmt, unsigned int ith, const void* _taskarg)
  {
    const TaskArg_t* taskarg = static_cast<const TaskArg_t*>(_taskarg);
    size_t count __attribute__((unused)) = 0;
   M4_PARAM(`count += TraitFormalParam$1::get_nparam(&taskarg->f$1);
    if (ith < count) return (kaapi_access_mode_t)TYPEMODE2VALUE<typename TraitFormalParam$1::mode_t>::value;
    ', ` ', `
    ')
    return KAAPI_ACCESS_MODE_VOID;
  }
  
  static void* get_off_param(const struct kaapi_format_t*, unsigned int ith, const void* _taskarg)
  {
    TaskArg_t* taskarg = static_cast<TaskArg_t*>((void*)_taskarg);
    size_t countp __attribute__((unused)) = 0, count __attribute__((unused)) = 0;
   M4_PARAM(`count += TraitFormalParam$1::get_nparam(&taskarg->f$1);
    if (ith < count) return (void*)TraitFormalParam$1::get_data(&taskarg->f$1, ith-countp);
    countp = count; 
    ', ` ', `
    ')
    return 0;
  }

  static kaapi_access_t get_access_param(const struct kaapi_format_t*, unsigned int ith, const void* _taskarg)
  {
    kaapi_access_t retval = {0, 0};
    const TaskArg_t* taskarg = static_cast<const TaskArg_t*>(_taskarg);
    size_t countp __attribute__((unused)) = 0, count __attribute__((unused)) = 0;
   M4_PARAM(`count += TraitFormalParam$1::get_nparam(&taskarg->f$1);
    if (ith < count) 
    {
      TraitFormalParam$1::get_access(&taskarg->f$1, ith-countp, &retval);
      return retval;
    }
    countp = count; 
    ', ` ', `
    ')
    retval.data    = 0;
    retval.version = 0;
    return retval;
  }
  
  static void set_access_param(const struct kaapi_format_t*, unsigned int ith, void* _taskarg, const kaapi_access_t* a)
  {
    TaskArg_t* taskarg = static_cast<TaskArg_t*>(_taskarg);
    size_t countp __attribute__((unused)) = 0, count __attribute__((unused)) = 0;
   M4_PARAM(`count += TraitFormalParam$1::get_nparam(&taskarg->f$1);
    if (ith < count) 
    {
      TraitFormalParam$1::set_access(&taskarg->f$1, ith-countp, a);
      return;
    }
    countp = count; 
    ', ` ', `
    ')
  }

  static const kaapi_format_t* get_fmt_param(const struct kaapi_format_t*, unsigned int ith, const void* _taskarg)
  {
    const TaskArg_t* taskarg = static_cast<const TaskArg_t*>(_taskarg);
    size_t count __attribute__((unused)) = 0;
   M4_PARAM(`count += TraitFormalParam$1::get_nparam(&taskarg->f$1);
    if (ith < count) return WrapperFormat<typename TraitFormalParam$1::type_t>::get_c_format();
    ', ` ', `
    ')
    return 0;
  }
  
  static kaapi_memory_view_t get_view_param( const struct kaapi_format_t*, unsigned int ith, const void* _taskarg )
  {
    const TaskArg_t* taskarg = static_cast<const TaskArg_t*>(_taskarg);
    size_t count __attribute__((unused)) = 0;
   M4_PARAM(`count += TraitFormalParam$1::get_nparam(&taskarg->f$1);
    if (ith < count) return TraitFormalParam$1::get_view_param(&taskarg->f$1, ith);
    ', ` ', `
    ')
    return kaapi_memory_view_t(); /* empty new view */
  }

  static void set_view_param( const struct kaapi_format_t*, unsigned int ith, void* _taskarg, const kaapi_memory_view_t* view )
  {
    TaskArg_t* taskarg = static_cast<TaskArg_t*>(_taskarg);
    size_t count __attribute__((unused)) = 0;
   M4_PARAM(`count += TraitFormalParam$1::get_nparam(&taskarg->f$1);
    if (ith < count) return TraitFormalParam$1::set_view_param(&taskarg->f$1, ith, view);
    ', ` ', `
    ')
  }

  static void reducor(const struct kaapi_format_t*, unsigned int ith, void* _taskarg, const void* value)
  {
    TaskArg_t* taskarg = static_cast<TaskArg_t*>(_taskarg);
    size_t countp __attribute__((unused)) = 0, count __attribute__((unused)) = 0;
   M4_PARAM(`count += TraitFormalParam$1::get_nparam(&taskarg->f$1);
    if (ith < count) { 
      TraitFormalParam$1::reducor_fnc(
          *(void**)TraitFormalParam$1::get_data(&taskarg->f$1, ith-count), value
      ); 
      return; 
    }
    countp = count;         
    ', ` ', `
    ')
  }

  static void redinit(const struct kaapi_format_t*, unsigned int ith, const void* _taskarg, void* value)
  {
    const TaskArg_t* taskarg = static_cast<const TaskArg_t*>(_taskarg);
    size_t count __attribute__((unused)) = 0;
   M4_PARAM(`count += TraitFormalParam$1::get_nparam(&taskarg->f$1);
    if (ith < count) { 
      TraitFormalParam$1::redinit_fnc(value); 
    }
    ', ` ', `
    ')
  }

  static void get_task_binding(const struct kaapi_format_t*, const void* _taskarg, kaapi_task_binding_t* tb)
  {
    /*const TaskArg_t* taskarg = static_cast<const TaskArg_t*>(_taskarg);*/
    tb->type = KAAPI_BINDING_ANY;
  }

  static kaapi_format_t* registerformat()
  {
    // here we assume no concurrency during startup calls of the library that initialize format objects
    static FormatTask task_fmt(
          typeid(TASK).name(),
          sizeof(TaskArg_t),
          &get_count_params,
          &get_mode_param,
          &get_off_param,
          &get_access_param,
          &set_access_param,
          &get_fmt_param,
          &get_view_param,
          &set_view_param,
          &reducor,
          &redinit,
          &get_task_binding
    );
#if defined(KAAPI_DEBUG)
    kaapi_fmt_set_dot_name( task_fmt.get_c_format(), TaskDOT<TASK>::name() );
    kaapi_fmt_set_dot_color( task_fmt.get_c_format(), TaskDOT<TASK>::color() );
#endif
      
    return task_fmt.get_c_format();
  }
};


// Top level class with main entry point
template<class TASK M4_PARAM(`,typename TraitFormalParam$1', `', ` ')>
struct KAAPI_FORMATCLOSURE(KAAPI_NUMBER_PARAMS) :
  public KAAPI_FORMATCLOSURE_SD(KAAPI_NUMBER_PARAMS)<
    TASK M4_PARAM(`,TraitFormalParam$1', `', ` '),
    false /* do not use static format for C++ interface */
/*
    KAAPI_TASKARG(KAAPI_NUMBER_PARAMS) ifelse(KAAPI_NUMBER_PARAMS,0,`',`<M4_PARAM(`TraitFormalParam$1', `', `,')>')::is_static
*/
  >
{
  static kaapi_bodies_t default_bodies;

  M4_PARAM(`typedef typename TraitFormalParam$1::type_inclosure_t inclosure$1_t;
  ', `', `')
  typedef KAAPI_TASKARG(KAAPI_NUMBER_PARAMS) ifelse(KAAPI_NUMBER_PARAMS,0,`',`<M4_PARAM(`TraitFormalParam$1', `', `,')>') TaskArg_t;
};


template<class TASK M4_PARAM(`,typename TraitFormalParam$1', `', ` ')>
struct KAAPI_INITFORMATCLOSURE(KAAPI_NUMBER_PARAMS) {

  M4_PARAM(`typedef typename TraitFormalParam$1::mode_t mode$1_t;
  ', `', `')
  M4_PARAM(`typedef typename TraitFormalParam$1::type_inclosure_t inclosure$1_t;
  ', `', `')
  M4_PARAM(`typedef typename TraitFormalParam$1::formal_t formal$1_t;
  ', `', `')
  M4_PARAM(`typedef typename TraitFormalParam$1::signature_t signature$1_t;
  ', `', `')

  /* GPU or CPU registration */
  /* trap function for undefined body */
  static kaapi_task_body_t registercpubody( kaapi_format_t* fmt, void (TaskBodyCPU<TASK>::*method)( THIS_TYPE_IS_USED_ONLY_INTERNALLY dummy M4_PARAM(`, signature$1_t', `', `') ) )
  {
    return 0;
  }

#if 0
  static kaapi_task_body_t registercpubody( kaapi_format_t* fmt, void (TaskBodyCPU<TASK>::*method)( M4_PARAM(`formal$1_t', `', `,') ) )
  {
    typedef void (TASK::*type_default_t)(THIS_TYPE_IS_USED_ONLY_INTERNALLY M4_PARAM(`, signature$1_t', `', `'));
    type_default_t f_default = &TASK::operator();
    if ((type_default_t)method == f_default) return 0;
    kaapi_task_body_t body = (kaapi_task_body_t)KAAPIWRAPPER_CPU_BODY(KAAPI_NUMBER_PARAMS)<1, TASK M4_PARAM(`,TraitFormalParam$1', `', ` ')>::body;
    kaapi_task_body_t bodywithhandle = (kaapi_task_body_t)KAAPIWRAPPER_CPU_BODY(KAAPI_NUMBER_PARAMS)<1, TASK M4_PARAM(`,TraitFormalParam$1', `', ` ')>::bodywithhandle;
    kaapi_format_taskregister_body(fmt, body, bodywithhandle, KAAPI_PROC_TYPE_CPU );
    return body;
  }

  static kaapi_task_body_t registercpubody( kaapi_format_t* fmt, void (TaskBodyCPU<TASK>::*method)( Thread* M4_PARAM(`, formal$1_t', `', `') ) )
  {
    typedef void (TASK::*type_default_t)(THIS_TYPE_IS_USED_ONLY_INTERNALLY M4_PARAM(`, signature$1_t', `', `'));
    type_default_t f_default = &TASK::operator();
    if ((type_default_t)method == f_default) return 0;

    kaapi_task_body_t body = (kaapi_task_body_t)KAAPIWRAPPER_CPU_BODY(KAAPI_NUMBER_PARAMS)<2, TASK M4_PARAM(`,TraitFormalParam$1', `', ` ')>::body;
    kaapi_task_body_t bodywithhandle = (kaapi_task_body_t)KAAPIWRAPPER_CPU_BODY(KAAPI_NUMBER_PARAMS)<2, TASK M4_PARAM(`,TraitFormalParam$1', `', ` ')>::bodywithhandle;
    kaapi_format_taskregister_body(fmt, body, bodywithhandle, KAAPI_PROC_TYPE_CPU );
    return body;
  }
#else // #if 0
  ifelse(KAAPI_NUMBER_PARAMS,0,`',`template< M4_PARAM(`class F$1_t', `', `,')>')
  static kaapi_task_body_t registercpubody( kaapi_format_t* fmt, void (TaskBodyCPU<TASK>::*method)( M4_PARAM(`F$1_t', `', `,') ) )
  {
    typedef void (TASK::*type_default_t)(THIS_TYPE_IS_USED_ONLY_INTERNALLY M4_PARAM(`, signature$1_t', `', `'));
    type_default_t f_default = &TASK::operator();
    if ((type_default_t)method == f_default) return 0;
    kaapi_task_body_t body = (kaapi_task_body_t)KAAPIWRAPPER_CPU_BODY(KAAPI_NUMBER_PARAMS)<1, TASK M4_PARAM(`,TraitFormalParam$1', `', ` ')>::body;
    kaapi_task_body_t bodywithhandle = (kaapi_task_body_t)KAAPIWRAPPER_CPU_BODY(KAAPI_NUMBER_PARAMS)<1, TASK M4_PARAM(`,TraitFormalParam$1', `', ` ')>::bodywithhandle;
    kaapi_format_taskregister_body(fmt, body, bodywithhandle, KAAPI_PROC_TYPE_CPU );
    return body;
  }

  ifelse(KAAPI_NUMBER_PARAMS,0,`',`template< M4_PARAM(`class F$1_t', `', `,')>')
  static kaapi_task_body_t registercpubody( kaapi_format_t* fmt, void (TaskBodyCPU<TASK>::*method)( Thread* M4_PARAM(`, F$1_t', `', `') ) )
  {
    typedef void (TASK::*type_default_t)(THIS_TYPE_IS_USED_ONLY_INTERNALLY M4_PARAM(`, signature$1_t', `', `'));
    type_default_t f_default = &TASK::operator();
    if ((type_default_t)method == f_default) return 0;

    kaapi_task_body_t body = (kaapi_task_body_t)KAAPIWRAPPER_CPU_BODY(KAAPI_NUMBER_PARAMS)<2, TASK M4_PARAM(`,TraitFormalParam$1', `', ` ')>::body;
    kaapi_task_body_t bodywithhandle = (kaapi_task_body_t)KAAPIWRAPPER_CPU_BODY(KAAPI_NUMBER_PARAMS)<2, TASK M4_PARAM(`,TraitFormalParam$1', `', ` ')>::bodywithhandle;
    kaapi_format_taskregister_body(fmt, body, bodywithhandle, KAAPI_PROC_TYPE_CPU );
    return body;
  }

  ifelse(KAAPI_NUMBER_PARAMS,0,`',`template< M4_PARAM(`class F$1_t', `', `,')>')
  static kaapi_task_body_t registercpubody( kaapi_format_t* fmt, void (TaskBodyCPU<TASK>::*method)( StaticSchedInfo* M4_PARAM(`, F$1_t', `', `') ) )
  {
    typedef void (TASK::*type_default_t)(THIS_TYPE_IS_USED_ONLY_INTERNALLY M4_PARAM(`, signature$1_t', `', `'));
    type_default_t f_default = &TASK::operator();
    if ((type_default_t)method == f_default) return 0;
    kaapi_task_body_t body = (kaapi_task_body_t)KAAPIWRAPPER_CPU_BODY(KAAPI_NUMBER_PARAMS)<4, TASK M4_PARAM(`,TraitFormalParam$1', `', ` ')>::body;
    kaapi_task_body_t bodywithhandle = (kaapi_task_body_t)KAAPIWRAPPER_CPU_BODY(KAAPI_NUMBER_PARAMS)<4, TASK M4_PARAM(`,TraitFormalParam$1', `', ` ')>::bodywithhandle;
    kaapi_format_taskregister_body(fmt, body, bodywithhandle, KAAPI_PROC_TYPE_CPU );
    return body;
  }

  ifelse(KAAPI_NUMBER_PARAMS,0,`',`template< M4_PARAM(`class F$1_t', `', `,')>')
  static kaapi_task_body_t registercpubody( kaapi_format_t* fmt, void (TaskBodyCPU<TASK>::*method)( const StaticSchedInfo* M4_PARAM(`, F$1_t', `', `') ) )
  {
    typedef void (TASK::*type_default_t)(THIS_TYPE_IS_USED_ONLY_INTERNALLY M4_PARAM(`, signature$1_t', `', `'));
    type_default_t f_default = &TASK::operator();
    if ((type_default_t)method == f_default) return 0;
    kaapi_task_body_t body = (kaapi_task_body_t)KAAPIWRAPPER_CPU_BODY(KAAPI_NUMBER_PARAMS)<4, TASK M4_PARAM(`,TraitFormalParam$1', `', ` ')>::body;
    kaapi_task_body_t bodywithhandle = (kaapi_task_body_t)KAAPIWRAPPER_CPU_BODY(KAAPI_NUMBER_PARAMS)<4, TASK M4_PARAM(`,TraitFormalParam$1', `', ` ')>::bodywithhandle;
    kaapi_format_taskregister_body(fmt, body, bodywithhandle, KAAPI_PROC_TYPE_CPU );
    return body;
  }

#endif

  static kaapi_task_body_t registercpubody( kaapi_format_t* fmt, void (TaskBodyCPU<TASK>::*method)( StealContext* M4_PARAM(`, formal$1_t', `', `') ) )
  {
    typedef void (TASK::*type_default_t)(THIS_TYPE_IS_USED_ONLY_INTERNALLY M4_PARAM(`, signature$1_t', `', `'));
    type_default_t f_default = &TASK::operator();
    if ((type_default_t)method == f_default) return 0;
    kaapi_task_body_t body = (kaapi_task_body_t)KAAPIWRAPPER_CPU_BODY(KAAPI_NUMBER_PARAMS)<3, TASK M4_PARAM(`,TraitFormalParam$1', `', ` ')>::body;
    kaapi_task_body_t bodywithhandle = (kaapi_task_body_t)KAAPIWRAPPER_CPU_BODY(KAAPI_NUMBER_PARAMS)<3, TASK M4_PARAM(`,TraitFormalParam$1', `', ` ')>::bodywithhandle;
    kaapi_format_taskregister_body(fmt, body, bodywithhandle, KAAPI_PROC_TYPE_CPU );
    return body;
  }

#if 0
  static kaapi_task_body_t registercpubody( kaapi_format_t* fmt, void (TaskBodyCPU<TASK>::*method)( StaticSchedInfo* M4_PARAM(`, formal$1_t', `', `') ) )
  {
    typedef void (TASK::*type_default_t)(THIS_TYPE_IS_USED_ONLY_INTERNALLY M4_PARAM(`, signature$1_t', `', `'));
    type_default_t f_default = &TASK::operator();
    if ((type_default_t)method == f_default) return 0;
    kaapi_task_body_t body = (kaapi_task_body_t)KAAPIWRAPPER_CPU_BODY(KAAPI_NUMBER_PARAMS)<4, TASK M4_PARAM(`,TraitFormalParam$1', `', ` ')>::body;
    kaapi_task_body_t bodywithhandle = (kaapi_task_body_t)KAAPIWRAPPER_CPU_BODY(KAAPI_NUMBER_PARAMS)<4, TASK M4_PARAM(`,TraitFormalParam$1', `', ` ')>::bodywithhandle;
    kaapi_format_taskregister_body(fmt, body, bodywithhandle, KAAPI_PROC_TYPE_CPU );
    return body;
  }

  static kaapi_task_body_t registercpubody( kaapi_format_t* fmt, void (TaskBodyCPU<TASK>::*method)( const StaticSchedInfo* M4_PARAM(`, formal$1_t', `', `') ) )
  {
    typedef void (TASK::*type_default_t)(THIS_TYPE_IS_USED_ONLY_INTERNALLY M4_PARAM(`, signature$1_t', `', `'));
    type_default_t f_default = &TASK::operator();
    if ((type_default_t)method == f_default) return 0;
    kaapi_task_body_t body = (kaapi_task_body_t)KAAPIWRAPPER_CPU_BODY(KAAPI_NUMBER_PARAMS)<4, TASK M4_PARAM(`,TraitFormalParam$1', `', ` ')>::body;
    kaapi_task_body_t bodywithhandle = (kaapi_task_body_t)KAAPIWRAPPER_CPU_BODY(KAAPI_NUMBER_PARAMS)<4, TASK M4_PARAM(`,TraitFormalParam$1', `', ` ')>::bodywithhandle;
    kaapi_format_taskregister_body(fmt, body, bodywithhandle, KAAPI_PROC_TYPE_CPU );
    return body;
  }
#endif

  /* GPU registration */

  /* trap function for undefined body */
  static kaapi_task_body_t registergpubody( kaapi_format_t* fmt, void (TaskBodyGPU<TASK>::*method)( THIS_TYPE_IS_USED_ONLY_INTERNALLY  M4_PARAM(`, signature$1_t', `', `') ) )
  {
    return 0;
  }

  static kaapi_task_body_t registergpubody( kaapi_format_t* fmt, void (TaskBodyGPU<TASK>::*method)( M4_PARAM(`formal$1_t', `', `,') ) )
  {
    typedef void (TASK::*type_default_t)(THIS_TYPE_IS_USED_ONLY_INTERNALLY M4_PARAM(`, signature$1_t', `', `'));
    type_default_t f_default = &TASK::operator();
    if ((type_default_t)method == f_default) return 0;
    kaapi_task_body_t body = (kaapi_task_body_t)KAAPIWRAPPER_GPU_BODY(KAAPI_NUMBER_PARAMS)<1, TASK M4_PARAM(`,TraitFormalParam$1', `', ` ')>::body;
    kaapi_task_body_t bodywithhandle = (kaapi_task_body_t)KAAPIWRAPPER_GPU_BODY(KAAPI_NUMBER_PARAMS)<1, TASK M4_PARAM(`,TraitFormalParam$1', `', ` ')>::bodywithhandle;
    kaapi_format_taskregister_body(fmt, body, bodywithhandle, KAAPI_PROC_TYPE_GPU );
    return body;
  }

  ifelse(KAAPI_NUMBER_PARAMS,0,`',`template<M4_PARAM(`typename formal_or_sig$1_t', `', `,') >')
  static kaapi_task_body_t registergpubody( kaapi_format_t* fmt, void (TaskBodyGPU<TASK>::*method)( gpuStream M4_PARAM(`, formal_or_sig$1_t', `', `') ) )
  {
    typedef void (TASK::*type_default_t)(THIS_TYPE_IS_USED_ONLY_INTERNALLY M4_PARAM(`, signature$1_t', `', `'));
    type_default_t f_default = &TASK::operator();
    if ((type_default_t)method == f_default) return 0;
    kaapi_task_body_t body = (kaapi_task_body_t)KAAPIWRAPPER_GPU_BODY(KAAPI_NUMBER_PARAMS)<2, TASK M4_PARAM(`,TraitFormalParam$1', `', ` ')>::body;
    kaapi_task_body_t bodywithhandle = (kaapi_task_body_t)KAAPIWRAPPER_GPU_BODY(KAAPI_NUMBER_PARAMS)<2, TASK M4_PARAM(`,TraitFormalParam$1', `', ` ')>::bodywithhandle;
    kaapi_format_taskregister_body(fmt, body, bodywithhandle, KAAPI_PROC_TYPE_GPU );
    return body;
  }

  /* Alpha value registration */

  /* trap function for undefined body */
  static kaapi_task_body_t registeralphabody( kaapi_format_t* fmt, void (TaskBodyAlpha<TASK>::*method)( THIS_TYPE_IS_USED_ONLY_INTERNALLY  M4_PARAM(`, signature$1_t', `', `') ) )
  {
    return 0;
  }

  ifelse(KAAPI_NUMBER_PARAMS,0,`',`template<M4_PARAM(`typename formal_or_sig$1_t', `', `,') >')
  static kaapi_task_body_t registeralphabody( kaapi_format_t* fmt, void (TaskBodyAlpha<TASK>::*method)( ka::pointer_w<float> M4_PARAM(`, formal_or_sig$1_t', `', `') ) )
  {
    typedef void (TASK::*type_default_t)(THIS_TYPE_IS_USED_ONLY_INTERNALLY M4_PARAM(`, signature$1_t', `', `'));
    type_default_t f_default = &TASK::operator();
    if ((type_default_t)method == f_default) return 0;
    kaapi_task_body_t body = (kaapi_task_body_t)KAAPIWRAPPER_ALPHA_BODY(KAAPI_NUMBER_PARAMS)<2, TASK M4_PARAM(`,TraitFormalParam$1', `', ` ')>::body;
//    kaapi_task_body_t bodywithhandle = (kaapi_task_body_t)KAAPIWRAPPER_ALPHA_BODY(KAAPI_NUMBER_PARAMS)<2, TASK M4_PARAM(`,TraitFormalParam$1', `', ` ')>::bodywithhandle;
    kaapi_format_taskregister_alphabody(fmt, body);
    return body;
  }

  ifelse(KAAPI_NUMBER_PARAMS,0,`',`template<M4_PARAM(`typename SIG$1_t', `', `,') >')
  static kaapi_bodies_t registerbodies( kaapi_format_t* fmt, void (TASK::*method)( Thread* thread M4_PARAM(`, SIG$1_t', `', `') ) )
  {
    kaapi_bodies_t retval = kaapi_bodies_t( registercpubody( fmt, &TaskBodyCPU<TASK>::operator() ), 
                                            registergpubody( fmt, &TaskBodyGPU<TASK>::operator() ),
					    registeralphabody( fmt, &TaskBodyAlpha<TASK>::operator() ) );
    return retval;
  }
};



template<class TASK M4_PARAM(`,typename TraitFormalParam$1', `', ` ')>
kaapi_bodies_t KAAPI_FORMATCLOSURE(KAAPI_NUMBER_PARAMS)<TASK M4_PARAM(`,TraitFormalParam$1', `', ` ')>::default_bodies =
    kaapi_bodies_t( KAAPI_INITFORMATCLOSURE(KAAPI_NUMBER_PARAMS)<TASK M4_PARAM(`,TraitFormalParam$1', `', ` ')>::registercpubody( KAAPI_FORMATCLOSURE(KAAPI_NUMBER_PARAMS)<TASK M4_PARAM(`,TraitFormalParam$1', `', ` ')>::registerformat(), &TaskBodyCPU<TASK>::operator() ), 
                    KAAPI_INITFORMATCLOSURE(KAAPI_NUMBER_PARAMS)<TASK M4_PARAM(`,TraitFormalParam$1', `', ` ')>::registergpubody( KAAPI_FORMATCLOSURE(KAAPI_NUMBER_PARAMS)<TASK M4_PARAM(`,TraitFormalParam$1', `', ` ')>::registerformat(), &TaskBodyGPU<TASK>::operator() ),
                    KAAPI_INITFORMATCLOSURE(KAAPI_NUMBER_PARAMS)<TASK M4_PARAM(`,TraitFormalParam$1', `', ` ')>::registeralphabody( KAAPI_FORMATCLOSURE(KAAPI_NUMBER_PARAMS)<TASK M4_PARAM(`,TraitFormalParam$1', `', ` ')>::registerformat(), &TaskBodyAlpha<TASK>::operator() ) );

