    template<M4_PARAM(`class E$1', `', `,')>
    void operator()( M4_PARAM(`E$1& e$1', `', `, ') )
    {
      _threadgroup->begin_partition();      
      TASKGENERATOR()( M4_PARAM(`e$1', `', `, ') );
      _threadgroup->end_partition();
      _threadgroup->execute();
    }
