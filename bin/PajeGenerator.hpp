#ifndef _VITE_PAJEGEN_
#define _VITE_PAJEGEN_

#include <string>

class  PajeGenerator {

protected :
    FILE*   _dest;         ; // File destination

public :

    //! Constructor
    PajeGenerator  ()                                    ;
    //! Destructor
    ~PajeGenerator ()                                    ;
    //! Open the file and write the first part of the trace 
    int initTrace (const std::string& name, int depth, 
                    int procNbr, int stateType, int eventType, int linkTypeNbr, int varNbr);
    //! Add a state to the trace
    void addState  (int proc    , int state, double time);
    //! Start a link on the trace
    void startLink (int proc    , int type , double time);
    //! End a link on the trace
    void endLink   (int proc    , int type , double time);
    //! Add an event to the trace
    void addEvent  (int proc    , int type , double time);
    //! Inc a counter to the trace
    void incCpt    (int proc    , int var  , double time);
    //! Dec a counter to the trace
    void decCpt    (int proc    , int var  , double time);
    //! End the container and close the file
    void endTrace  ()                                    ;


};


#endif


