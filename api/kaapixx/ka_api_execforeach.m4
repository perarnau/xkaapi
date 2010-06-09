    template<M4_PARAM(`class E$1', `', `,')>
    void operator()( M4_PARAM(`E$1& e$1', `', `, ') )
    {
      if (_beg == _end) return;
      _threadgroup->begin_partition();
      tpart = kaapi_get_elapsedtime();
      TASKGENERATOR()( M4_PARAM(`e$1', `', `, ') );
      tpart = kaapi_get_elapsedtime()-tpart;
      _threadgroup->end_partition();
      _threadgroup->save();
      while (_beg != _end)
      {
        t0 = kaapi_get_elapsedtime();
        _threadgroup->execute();
        t1 = kaapi_get_elapsedtime();
        if (step >0) total += t1-t0;
        std::cout << step << ":: Time: " << t1 - t0 << std::endl;
        ++step;
        if (++_beg != _end) _threadgroup->restore();
      }
    }