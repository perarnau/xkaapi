
/* Task Interface for debug information using dot graph */
template<class TASK>
struct TaskDOT: public TASK {
      /* DOT name to display */
    static const char* name() { return 0; }

     /* DOT color name to display */
    static const char* color() { return 0; }
};

