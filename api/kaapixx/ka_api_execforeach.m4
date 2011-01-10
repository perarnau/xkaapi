    template<M4_PARAM(`class E$1', `', `,')>
    void operator()( M4_PARAM(`E$1& e$1', `', `, ') )
    {
        if (_beg == _end) return;
        _threadgroup->begin_partition();
        tpart = kaapi_get_elapsedtime();
        TASKGENERATOR()( M4_PARAM(`e$1', `', `, ') );
        //_threadgroup->print();    
        _threadgroup->end_partition();
        tpart = kaapi_get_elapsedtime()-tpart;
        _threadgroup->save();
#if 1
        t0 = kaapi_get_elapsedtime();
#endif
        while (_beg != _end)
        {
#if 1
          t0 = kaapi_get_elapsedtime();
#endif
          _threadgroup->start_execute();
          _threadgroup->wait_execute();
#if 1
          t1 = kaapi_get_elapsedtime();
          if (step >0) total += t1-t0;
#endif
#if 1
          std::cout << step << ":: Time: " << t1 - t0 << std::endl;
#endif          
          ++step;
          if (++_beg != _end) _threadgroup->restore();
        }
#if 1
        t1 = kaapi_get_elapsedtime();
#endif
#if 1
        std::cout << ":: ForEach #loops: " << step << ", total time (except first iteration):" << total
                  << ", average:" << total / (step-1) << ", partition step:" << tpart << std::endl;
#else
        std::cout << ":: ForEach #loops: " << step << ", total time (except first iteration):" << total
                  << ", average:" << (t1-t0) / (step) << ", partition step:" << tpart << std::endl;
#endif
        _threadgroup->end_execute(); /* free data structure */
    }
