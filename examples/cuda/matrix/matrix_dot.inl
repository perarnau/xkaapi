
template<>
struct TaskDOT<TaskDGEMM> {
    static const char* name() { return "DGEMM"; }
    static const char* color() { return "grey"; }
};

template<>
struct TaskDOT<TaskDPOTRF> {
    static const char* name() { return "DPOTRF"; }
    static const char* color() { return "orange"; }
};

template<>
struct TaskDOT<TaskDTRSM> {
    static const char* name() { return "DTRSM"; }
    static const char* color() { return "skyblue"; }
};

template<>
struct TaskDOT<TaskDSYRK> {
    static const char* name() { return "DSYRK"; }
    static const char* color() { return "green"; }
};

template<>
struct TaskDOT<TaskDGETRF> {
    static const char* name() { return "DGETRF"; }
    static const char* color() { return "orange"; }
};

template<>
struct TaskDOT<TaskPlasmaDGESSM> {
    static const char* name() { return "DGESSM"; }
    static const char* color() { return "orange"; }
};

template<>
struct TaskDOT<TaskPlasmaDTSTRF> {
    static const char* name() { return "DTSTRF"; }
    static const char* color() { return "orange"; }
};

template<>
struct TaskDOT<TaskPlasmaDSSSSM> {
    static const char* name() { return "DSSSSM"; }
    static const char* color() { return "orange"; }
};

