
    /* KAAPI_NUMBER_PARAMS is the number of possible parameters */
    template<M4_PARAM(`class E$1', `', `,')>
    RFO::Closure* operator()( M4_PARAM(`const E$1& e$1', `', `,') )
    {
      return KAAPI_INITFORMATCLOSURE(KAAPI_NUMBER_PARAMS)::doit(_thread, _attr, &TASK::operator(), M4_PARAM(`e$1', `', `,') );
    }
