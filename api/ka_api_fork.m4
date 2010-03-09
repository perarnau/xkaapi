
    /* KAAPI_NUMBER_PARAMS is the number of possible parameters */
    template<class SIGNATURE, M4_PARAM(`class E$1, class F$1', `', `, ')>
    kaapi_task_t* KAAPI_NAME(PushArg,KAAPI_NUMBER_PARAMS)( void (SIGNATURE::*)( Thread* M4_PARAM(`, F$1', `', `') ), M4_PARAM(`const E$1& e$1', `', `,') )
    {
      M4_PARAM(`typedef typename TraitUAMParam<F$1>::uamttype_t uamttype$1_t;
      ', `', `')
      M4_PARAM(`typedef typename uamttype$1_t::template UAMParam<TYPE_INTASK>::type_t inclosure$1_t;
      ', `', `')
      typedef KAAPI_TASKARG(KAAPI_NUMBER_PARAMS) ifelse(KAAPI_NUMBER_PARAMS,0,`',`<M4_PARAM(`uamttype$1_t', `', `,')>') TaskArg_t;

      typedef KAAPI_FORMATCLOSURE(KAAPI_NUMBER_PARAMS)<TASK, M4_PARAM(`TraitUAMParam<F$1> ', `', `,')> KaapiFormatTask_t;
      
#if !defined(KAAPI_NDEBUG)
      M4_PARAM(`typedef typename TraitUAMParam<F$1>::mode_t mode_formal$1_t;
      ', `', `')
      M4_PARAM(`typedef typename TraitUAMParam<E$1>::mode_t mode_effective$1_t;
      ', `', `')
      M4_PARAM(`WARNING_UNDEFINED_PASSING_RULE<mode_effective$1_t, mode_formal$1_t, FOR_ARG<$1>, FOR_TASKNAME<TASK> >::IS_COMPATIBLE();
      ', `', `')
#endif
      kaapi_task_t* clo = kaapi_stack_toptask( _stack);
      kaapi_task_initdfg( clo, KaapiFormatTask_t::bodyid.value, kaapi_stack_pushdata(_stack, sizeof(TaskArg_t)) );
      TaskArg_t* arg = kaapi_task_getargst( clo, TaskArg_t);
      /* here we do not detect a compile time the error without compilation with -DKAAPI_DEBUG 
         todo -> grep a type in UAMTYpe with Effective type in parameter in place of actual inclosure
      */
      M4_PARAM(`new (&arg->f$1) inclosure$1_t(e$1);
      ', `', `')
      return clo;
    }

    template<M4_PARAM(`class E$1', `', `,')>
    kaapi_task_t* operator()( M4_PARAM(`const E$1& e$1', `', `, ') )
    {
      kaapi_task_t* clo = 
      KAAPI_NAME(PushArg,KAAPI_NUMBER_PARAMS)(
         &TASK::dummy_method_to_have_formal_param_type, M4_PARAM(`e$1', `', `, ') 
      );
      _attr(_stack, clo );
      kaapi_stack_pushtask( _stack);
      return clo; 
    }
