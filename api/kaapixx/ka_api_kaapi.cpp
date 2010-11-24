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
#include "kaapi_impl.h"
#include "kaapi++"
#include "ka_parser.h"
#include "ka_component.h"

namespace ka {

SetStickyC SetSticky;
SetStack SetInStack;
SetHeap SetInHeap;
SetLocalAttribut SetLocal;
DefaultAttribut SetDefault;
FlagReplyHead ReplyHead;
FlagReplyTail ReplyTail;


// --------------------------------------------------------------------
int System::saved_argc = 0;
char** System::saved_argv = 0;

// --------------------------------------------------------------------
Community::Community( const Community& com )
{ }


// --------------------------------------------------------------------
void Community::commit()
{
}


// --------------------------------------------------------------------
bool Community::is_leader() const
{ 
  return true;
}


// --------------------------------------------------------------------
void Community::leave() 
{ 
  Sync();
  kaapi_finalize();
}


// --------------------------------------------------------------------
Community System::initialize_community( int& argc, char**& argv )
  throw (RuntimeError, RestartException, ServerException)
{
  static bool is_called = false; if (is_called) return Community(0); is_called = true;
  
  /** Init should have been called by InitKaapiCXX
  */
  kaapi_assert(kaapi_init() == EALREADY );
  
  System::saved_argc = argc;
  System::saved_argv = new char*[argc];
  for (int i=0; i<argc; ++i)
  {
    int lenargvi = strlen(argv[i]);
    System::saved_argv[i] = new char[lenargvi+1];
    memcpy(System::saved_argv[i], argv[i], lenargvi );
    System::saved_argv[i][lenargvi] = 0;
  }
  
  /* first initialize KaapiComponentManager::prop from file $HOME/.kaapirc */
  std::string filename;
  char* name = getenv("HOME");
  if (name !=0) 
  {
    try {
      filename = name;
      filename = filename + "/.kaapirc";
      KaapiComponentManager::prop.load( filename );
    } catch (const IOError& e) { 
    } catch (...) { 
    }
  }

  /* second: initialize kaapi_prop from file $PWD/.kaapirc */
  name = getenv("PWD");
  if (name !=0) {
    try {
      filename = name;
      filename = filename + "/.kaapirc";
      KaapiComponentManager::prop.load( filename );
    } catch (const IOError& e) { 
    } catch (...) { 
    }
  }

  if (KaapiComponentManager::initialize( argc, argv ) != 0)
    Exception_throw( RuntimeError("[ka::System::initialize], Kaapi not initialized") );

  return Community(0);
}


// --------------------------------------------------------------------
Community System::join_community( int& argc, char**& argv )
  throw (RuntimeError, RestartException, ServerException)
{
  Community thecom = System::initialize_community( argc, argv);
  thecom.commit();
  return thecom;
}


// --------------------------------------------------------------------
Community System::join_community( )
  throw (RuntimeError, RestartException, ServerException)
{
  int argc = 1;
  const char* targv[] = {"kaapi"};
  char** argv = (char**)targv;
  Community thecom = System::initialize_community( argc, argv);
  thecom.commit();
  return thecom;
}


// --------------------------------------------------------------------
void System::terminate()
{
  KaapiComponentManager::terminate();
}


// --------------------------------------------------------------------
int System::getRank()
{
  return 1;
}


// --------------------------------------------------------------------
void Sync()
{
  kaapi_sched_sync( );
}


SyncGuard::SyncGuard() : _thread( kaapi_self_thread() )
{
  kaapi_thread_save_frame( _thread, &_frame );
}

SyncGuard::~SyncGuard()
{
  kaapi_sched_sync( );
  kaapi_thread_restore_frame( _thread, &_frame );
}

} // namespace ka
