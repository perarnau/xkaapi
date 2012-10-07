/*
** xkaapi
** 
**
** Copyright 2009,2010,2011,2012 INRIA.
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
#ifndef _ATHA_INIT_H
#define _ATHA_INIT_H

#include "ka_types.h"
#include "ka_properties.h"
#include "ka_parser.h"
#include "ka_component.h"

namespace ka {

class Init : public KaapiComponent {
public:
  /** The component object for the util package
  */
  static Init component;

  /** Global variable = true iff verbose mode if on for Util library level
      Default value is false.
  */
  static bool verboseon;

  /** Global variable = true iff trace logging is enable
      Default value is false.
  */
  static bool enable_trace;

  /** The global id of this process
  */
  static uint32_t local_gid;

  /** Should be defined for internal purpose only...
  */
  static bool on_term;
  static bool on_thread;
  
  /** Change the current local_gid
  */
  static void set_local_gid( GlobalId newgid );
  
  /** Default constructor: assign the most prioritary number (0) for this component
  */
  Init();

  /** Initialize the default attributs for Mutex, Condition and Thread
      Should be called before any use of Util functions or methods.
      \return the error code
  */
  int initialize() throw();

  /** Terminate the use of the Util library
      \return the error code
  */
  int terminate() throw();

  /** Add the options for this Kaapi' Component
  */
  void add_options( Parser* parser, Properties* global_prop );

  /** Declare dependencies with other component
  */
  void declare_dependencies();
};


} // namespace Init
#endif
