
    /* KAAPI_NUMBER_PARAMS is the number of possible parameters */
    template<class TASKORSIG, M4_PARAM(`class E$1, class F$1', `', `, ')>
    kaapi_task_t* KAAPI_NAME(PushArg,KAAPI_NUMBER_PARAMS)( void (TASKORSIG::*)( M4_PARAM(`F$1', `', `, ') ), M4_PARAM(`E$1 e$1', `', `,') )
    {
      typedef KAAPI_CLOSURE(KAAPI_NUMBER_PARAMS)<TASK, M4_PARAM(`F$1', `', `,')> KaapiClosure;

/*
      M4_PARAM(`PassingRule<typename Trait_ParamClosure<E$1>::mode, typename Trait_ParamClosure<F$1>::mode, 
                ARG<$1>, FOR_TASKNAME<TASKORSIG> >::IS_COMPATIBLE();
      ', `', `')
*/

      kaapi_task_t* clo = kaapi_stack_toptask( _stack);
      kaapi_task_initdfg( _stack, clo, &KaapiClosure::body_cpu, kaapi_stack_pushdata(_stack, sizeof(KaapiClosure)) );
      /* this function call is the only way I currently found to register the format of the task, 
         idealy it should not be call and the clo->format should not be set at all.
      */
      clo->format = KaapiClosure::registerformat();
      KaapiClosure* arg = kaapi_task_getargst( clo, KaapiClosure);

      M4_PARAM(`Trait_ParamClosure<F$1>::link(arg->f$1, e$1);
      ', `', `')
      return clo;
    }

    template<M4_PARAM(`class E$1', `', `,')>
    void operator()( M4_PARAM(`E$1 e$1', `', `, ') )
    {
      kaapi_task_t* clo = KAAPI_NAME(PushArg,KAAPI_NUMBER_PARAMS)( &TASK::operator(), M4_PARAM(`e$1', `', `, ') );
      _attr(_stack, clo );
      kaapi_stack_pushtask( _stack);    
    }
