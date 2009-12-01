/* KAAPI public interface */
/*
** xkaapi
** 
** Created on Tue Mar 31 15:19:14 2009
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
#ifndef _ATHA_COMPONENT_H_
#define _ATHA_COMPONENT_H_

#include "kaapi.h"
#include "atha_parser.h"

namespace atha {

// -------------------------------------------------------------------------
/** KaapiComponent
    \ingroup atha
    A component if part of the library that could should be initialized, terminated
    and has some options passed with a Parser::Module object and setup from command
    line or file.
    An object of class KaapiComponent represents this part of library. Such object should
    be allocated for every part of the library and registered to the KaapiComponentManager.
    The KaapiComponentManager initialize and terminate only once each Component object.
    The dependencies between initializations are managed using a priority number (0 is
    the most prioritary number).
*/
class KaapiComponent {
public:
  /** Default cstor
  */
  KaapiComponent( const std::string& name, int priority );
  
  /** Default dstor
  */
  virtual ~KaapiComponent();
  
  /** Declare to use the component and all sub component
      This method may use require protected' method in order to
      record requirement between components.
  */
  virtual void declare_dependencies() =0;

  /** Initialize the default attributs for the component
      Should be called before any use of functions or methods on the component.
      Return 0 if no error occurs.
      \return the error code if !=0
  */
  virtual int initialize() throw();

  /** Commit
      Called when all components have been initialized
      Default implementation return 0.
      Return 0 if no error occurs.
      \return the error code if !=0
  */
  virtual int commit() throw();

  /** Terminate the use of the Util library
      Return 0 if no error occurs.
      \return the error code if !=0
  */
  virtual int terminate() throw() = 0;
  
  /** Give to the component the ability to register its options
      This method is usually implement by calling parser.add_module(m, p) on
      all modules that is declared by the component. The global property
      table is generally used to store option for an internal module.
      \param parser is the parser that will be used 
      \param global_prop is the global properties table where to store options
  */
  virtual void add_options( Parser* parser, Properties* global_prop );

  /** Inform that 'this' component is required
  */
  void is_required_by( KaapiComponent* caller = 0);
  
  /**
  */
  int get_priority() const;

  /**
  */
  bool is_registered() const;

  /**
  */
  void set_registered();

  /**
  */
  double startup_time() const;

protected:
  double       _startup_time;
  std::string  _name;
  int          _refcount;
  int          _priority;
  bool         _isregistered;

private:
  bool         _is_init;
  friend class KaapiComponentManager;
};


// -------------------------------------------------------------------------
/** List of all components registered for execution of this process
    \ingroup Util
*/
class KaapiComponentManager {
public:  
  /** List of priority for initialize module of KAAPI middleware
  */
  enum {
    UTIL_COMPONENT_PRIORITY     = 0,
    NETWORK_COMPONENT_PRIORITY  = 100,
    NS_COMPONENT_PRIORITY       = 110,
    SOCKNET_COMPONENT_PRIORITY  = 150,
    UDPNET_COMPONENT_PRIORITY   = 170,
    TAKTUKNET_COMPONENT_PRIORITY= 190,
    MANET_COMPONENT_PRIORITY    = 200,
    FDNET_COMPONENT_PRIORITY    = 210,
    MYNET_COMPONENT_PRIORITY    = 220,
    NETSTAT_COMPONENT_PRIORITY  = 250,
    NETDATA_COMPONENT_PRIORITY  = 260,
    NETCTRL_COMPONENT_PRIORITY  = 270,
    KERNEL_COMPONENT_PRIORITY   = 300,
    RFO_COMPONENT_PRIORITY      = 400,
    DFG_COMPONENT_PRIORITY      = 500,
    WS_COMPONENT_PRIORITY       = 600,
    ST_COMPONENT_PRIORITY       = 700,
    FT_COMPONENT_PRIORITY       = 800,
    DL_COMPONENT_PRIORITY       = 900
  };
  
  /** Defaut properties for the processus
      This table is initialized on return to the call to Util::KaapiComponentManager::initialize
      and contains definitions of values for the key definied for the 
      parser Util::KaapiComponentManager::parser.
      It could be used by all KaapiComponent during the invocation to initialize methods.
  */
  static Properties prop;
  
  /** Default parser
  */
  static Parser parser;

  /** Register a component object
      Return true iff the object has been registered for the first time
  */
  static bool register_object( KaapiComponent* obj );

  /** Register a component object from a shared library
      The caller should initialize the component and call commit.
      The invocation of terminate will be automatically called at the end the execution.
      Return true iff the object has been registered for the first time
  */
  static bool register_shlib_object( KaapiComponent* obj );
  
  /** Only do parsing of arguments and property files in order to set up prop table
  */
  static int parseoption(int& argc , char**& argv ) throw();

  /** Main entry point to initialize all the library
      Call parseoption if not already called.
  */
  static int initialize(int& argc , char**& argv ) throw();

  /** Main entry point to terminate the library
  */
  static int terminate() throw();
};


// -------------------------------------------------------------------------
/*
 * inline definition
 */
// -------------------------------------------------------------------------
inline int KaapiComponent::get_priority() const
{ return _priority; }

inline bool KaapiComponent::is_registered() const
{ return _isregistered; }

inline void KaapiComponent::set_registered()
{ _isregistered = true; }

inline double KaapiComponent::startup_time() const
{ return _startup_time; }

} // namespace Init
#endif
