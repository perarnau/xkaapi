    template<M4_PARAM(`class E$1', `', `,')>
    void operator()( M4_PARAM(`E$1& e$1', `', `, ') )
    {
        if (this->_beg == this->_end) return;
        this->prologue();
        TASKGENERATOR()( M4_PARAM(`e$1', `', `, ') );
        this->epilogue();
    }
