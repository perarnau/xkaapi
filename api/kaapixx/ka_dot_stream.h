/*
** xkaapi
** 
**
** Copyright 2009 INRIA.
**
** Contributors :
**
** thierry.gautier@inrialpes.fr
** 
** This software is a computer program whose purpose is to execute
** multithreaded computation with data flow synchronization between
** threads.
** 
** This software is governed by the CeCILL-C license under French law
** and abiding by the rules of distribution of free software.  You can
** use, modify and/ or redistribute the software under the terms of
** the CeCILL-C license as circulated by CEA, CNRS and INRIA at the
** following URL "http://www.cecill.info".
** 
** As a counterpart to the access to the source code and rights to
** copy, modify and redistribute granted by the license, users are
** provided only with a limited warranty and the software's author,
** the holder of the economic rights, and the successive licensors
** have only limited liability.
** 
** In this respect, the user's attention is drawn to the risks
** associated with loading, using, modifying and/or developing or
** reproducing the software by the user in light of its specific
** status of free software, that may mean that it is complicated to
** manipulate, and that also therefore means that it is reserved for
** developers and experienced professionals having in-depth computer
** knowledge. Users are therefore encouraged to load and test the
** software's suitability as regards their requirements in conditions
** enabling the security of their systems and/or data to be ensured
** and, more generally, to use and operate it in the same conditions
** as regards security.
** 
** The fact that you are presently reading this means that you have
** had knowledge of the CeCILL-C license and that you accept its
** terms.
** 
*/
#ifndef _KAAPI_DOTSTREAM_H_
#define _KAAPI_DOTSTREAM_H_

#include "ka_parser.h"

namespace ka {

// -------------------------------------------------------------------------
/** \brief The Dotifiable class defines interfaces to dump a dot graph representation of the object.
    \ingroup RFO
*/
class Dotifiable {
public:
  /** Virtual destructor
   */
  virtual ~Dotifiable();

  /** Dump a representation the object state into the dot stream. 
      Dot stream may be configured by properties (see DotStream class)
      in order to choose a representation of basic object.
   */
  virtual void dump ( ODotStream& o ) const =0;
};


// -------------------------------------------------------------------------
/** \brief DotStream to dump dot representation of Data Flow Graph
    \ingroup RFO
    The ODotStream is a wrapper object that allows to insert objects to
    define element of task's graph such as Graph, SubGraph, Task, AccessLink
    and OrderLink. Output is the dot language and some attributs for graphical
    representation may be specify using properties file, or the parsing
    of command line options.
    
    The graphical representation is printed into two parts: the first one is the
    prologue the allows to display beginning of the dot declaration (let us recall 
    that dot is an language with bloc definition that corresponds to diagraph 
    and subgraph definitions); The second part is the epilogue that allows to
    end the declaration. The prologue is printed out when the element of the
    task's graph is inserted into the ODotStream. The epilogue is printed out when
    the destructor of the task's graph element is called.
    
    For instance the follwing code:
    \code
    RFO::ODotStream sdot ( ... );
    
    { // if the begining of the scope of definition of temporary object
      sdot << RFO::ODotStream::Graph("test");
      sdot << RFO::ODotStream::Task(closure1, "Writer" );
      sdot << RFO::ODotStream::Task(closure2, "Reader" );
      sdot << RFO::ODotStream::OrderLink(closure1, closure2);
    } 
    // here temporary object should be been destroy in the following order:
    //     - the OrderLink object
    //     - the 2nd Task object
    //     - the 1st Task object
    //     - the Graph object
    \endcode
    \see DFG::ODotStream
*/
class ODotStream {
public:
  /** Cstor 
  */
  ODotStream( std::ostream* output )
   : _output(output)
  { }

  /** Dstor 
  */
  ~ODotStream() {}
  
  /** Return the module to parser argument
  */
  static ka::Parser::Module* get_module();
  static void terminate_module();

  /** Basic Object to be dumped into the stream
  */
  struct Object {
    Object( const std::string& n ) : name(n), output(0) {}
    virtual ~Object();
    virtual void prologue( std::ostream& o ) const;    
    virtual void epilogue( std::ostream& o ) const;    
    std::string   name;
    mutable std::ostream* output;
  };
  
  /** Attributs 
      * dot.graph.size="8.5,11.00" by default
  */
  struct Graph : public Object {
    Graph( const std::string& n ) : Object(n) {}
    ~Graph();
    void prologue( std::ostream& o ) const;    
    void epilogue( std::ostream& o ) const;    
  };

  /** Attributs 
  */
  struct SubGraph : public Object {
    SubGraph( void* a, const std::string& n ) : Object(n), address(a) {}
    ~SubGraph();
    void prologue( std::ostream& o ) const;
    void epilogue( std::ostream& o ) const;
    void* address;
  };

  /** Task 
  */
  struct Task : public Object {
    Task( const kaapi_task_t* a, const std::string& n, const std::string& s ="default" ) : Object(n), closure(a), shape(s) {}
    ~Task();
    void prologue( std::ostream& o ) const;    
    void epilogue( std::ostream& o ) const;    
    const kaapi_task_t* closure;
    const std::string& shape;
  };

 
  /** OrderLink 
  */
  struct OrderLink : public Object {
    OrderLink( const kaapi_task_t* c1, const kaapi_task_t* c2 ) : Object(""), closure1(c1), closure2(c2) {}
    ~OrderLink();
    void prologue( std::ostream& o ) const;    
    void epilogue( std::ostream& o ) const;    
    const kaapi_task_t* closure1;
    const kaapi_task_t* closure2;
  };

  /** OrderLink 
  */
  struct Text : public Object {
    Text( const std::string& s ) : Object(s) {}
    ~Text();
    void prologue( std::ostream& o ) const;
    void epilogue( std::ostream& o ) const;
  };

 /** Direction for AccessLink
 */
  struct Dir {
    enum Mode {
      forward,
      reverse,
      reversedot,
      both
    };
  };

  /** Access 
  */
  struct AccessLink : public RFO::ODotStream::Object {
    AccessLink( const kaapi_access_t* a, const kaapi_task_t* c, Dir::Mode m ) 
     : RFO::ODotStream::Object(""), access(a), closure(c), mode(m) 
    {}
    ~AccessLink();
    void prologue( std::ostream& o ) const;
    void epilogue( std::ostream& o ) const;
    const kaapi_access_t* access;
    const kaapi_task_t*   closure;
    Dir::Mode             mode;
  };

  /** Sync 
  */
  struct SyncLink : public RFO::ODotStream::Object {
    SyncLink( const kaapi_task_t* a, const kaapi_task_t* c ) 
     : RFO::ODotStream::Object(""), c1(a), c2(c)
    {}
    ~SyncLink();
    void prologue( std::ostream& o ) const;
    void epilogue( std::ostream& o ) const;
    const kaapi_task_t*  c1;
    const kaapi_task_t*  c2;
  };

  /** Encapsulation  
  */
  struct OriginalLink : public RFO::ODotStream::Object {
    OriginalLink( const kaapi_task_t* a, const kaapi_task_t* c ) 
     : RFO::ODotStream::Object(""), c1(a), c2(c)
    {}
    ~OriginalLink();
    void prologue( std::ostream& o ) const;
    void epilogue( std::ostream& o ) const;
    const kaapi_task_t*  c1;
    const kaapi_task_t*  c2;
  };

  /** Version
   */
  struct VersionLink : public RFO::ODotStream::Object {
    VersionLink( const void* d, int v, constkaapi_task_t* c, Dir::Mode m ) 
      : RFO::ODotStream::Object(""), gd(d), version(v), closure(c), mode(m) 
    {}
    
    ~VersionLink();
    void prologue( std::ostream& o ) const;
    void epilogue( std::ostream& o ) const;
    const void* gd;
    int version;
    const kaapi_task_t*  closure;
    Dir::Mode            mode;
  };

  /** output and object
  */
  ODotStream& operator<< (const Object& obj );
  
  /* Return std output stream 
  */
  std::ostream& get_stdout();
protected:
  std::ostream* _output;
};

inline std::ostream& ODotStream::get_stdout()
{ return *_output; }

} // namespace RFO

#endif
