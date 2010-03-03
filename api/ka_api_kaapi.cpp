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
#include "kaapi++"
#include "kaapi_impl.h"
#include "ka_parser.h"
#include "ka_component.h"

namespace ka {

SetStickyC SetSticky;
SetStack SetInStack;
SetHeap SetInHeap;
SetLocalAttribut SetLocal;
DefaultAttribut SetDefault;

/* save in init, used to push retn in leave or terminate */
static int main_retn_pushed = 0;
static kaapi_frame_t main_frame;

// --------------------------------------------------------------------------
std::string get_localhost()
{
  /* get canonical name from the syscall gethostname */
  std::string hostname;
  char buffer[1024];
  if ( gethostname( buffer, 1024 ) != 0) 
  {
    hostname = "localhost";
  } else {
   hostname = buffer;
  }
  return hostname;
}

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
  if (!main_retn_pushed)
  {
    /* push marker of the frame: retn */
    kaapi_stack_pushretn(kaapi_self_stack(), &main_frame);
    main_retn_pushed = 1;
  }

  Sync();
}


// --------------------------------------------------------------------
Community System::initialize_community( int& argc, char**& argv )
  throw (RuntimeError, RestartException, ServerException)
{
  static bool is_called = false; if (is_called) return Community(0); is_called = true;
  
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

  /* push marker of the frame: retn */
  kaapi_stack_save_frame( kaapi_self_stack(), &main_frame);
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
void System::terminate()
{
  if (!main_retn_pushed)
  {
    /* push marker of the frame: retn */
    kaapi_stack_pushretn(kaapi_self_stack(), &main_frame);
    main_retn_pushed = 1;
  }
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
  kaapi_sched_sync( kaapi_self_stack() );
}

} // namespace ka
