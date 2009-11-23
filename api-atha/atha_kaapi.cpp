// =========================================================================
// (c) INRIA, projet MOAIS, 2009
// Author: T. Gautier, X-Kaapi port
//
// =========================================================================
#include "athapascan-1"
#include "kaapi_impl.h"
#include "atha_parser.h"
#include "atha_component.h"


// autotools can prompt for this ...
const char* get_kaapi_version()
{ return XKAAPI_NAME"-"XKAAPI_VERSION; }

namespace a1 {

#if 0 /*TODO*/
const SpaceCollectionFormat SpaceCollectionFormat::theformat;
const SpaceCollectionFormat* const SpaceCollectionFormat::format  = &SpaceCollectionFormat::theformat;
#endif

SetStickyC SetSticky;
SetStack SetInStack;
SetHeap SetInHeap;
SetLocalAttribut SetLocal;
DefaultAttribut SetDefault;

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
}


// --------------------------------------------------------------------
Community System::initialize_community( int& argc, char**& argv )
  throw (RuntimeError, RestartException, ServerException)
{
  static bool is_called = false; if (is_called) return Community(0); is_called = true;
#if defined(KAAPI_USE_PAPI)
  /*if ajout pour papi*/
  int Events[NUM_EVENTS] = {PAPI_TOT_INS};
  long_long values[NUM_EVENTS];
  if (PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT)
  {
     perror("Message of PAPI_library_init");
     exit(1);
  }
  if (PAPI_thread_init(pthread_self) != PAPI_OK)
  {
     perror("Message of PAPI_thread_init");
     exit(1);
  }
  if (PAPI_start_counters(Events, NUM_EVENTS) != PAPI_OK)
  {
     perror("Message of PAPI_start_counters");
     exit(1);
  }
  if (PAPI_accum_counters(values, NUM_EVENTS) != PAPI_OK) // reset counter
  {
     perror("Message of PAPI_accum_counters");
     exit(1);
  }
#endif  
  atha::Parser::Module a1_module("a1");

  /* initialization of options for athapascan */
  a1_module.add_option("verbose", "false", "true iff verbose mode is activated");
  atha::KaapiComponentManager::parser.add_module( &a1_module, &atha::KaapiComponentManager::prop);
  
  /* first initialize KaapiComponentManager::prop from file $HOME/.a1rc */
  std::string filename;
  char* name = getenv("HOME");
  if (name !=0) 
  {
    try {
      filename = name;
      filename = filename + "/.a1rc";
      atha::KaapiComponentManager::prop.load( filename );
    } catch (const IOError& e) { 
    } catch (...) { 
    }
  }

  /* second: initialize kaapi_prop from file $PWD/.a1rc */
  name = getenv("PWD");
  if (name !=0) {
    try {
      filename = name;
      filename = filename + "/.a1rc";
      atha::KaapiComponentManager::prop.load( filename );
    } catch (const IOError& e) { 
    } catch (...) { 
    }
  }

  if (atha::KaapiComponentManager::initialize( argc, argv ) != 0)
    atha::Exception_throw( RuntimeError("[a1::System::initialize], Kaapi not initialized") );

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
  atha::KaapiComponentManager::terminate();
}


// --------------------------------------------------------------------
int System::getRank()
{
  return 1;
}


// --------------------------------------------------------------------
void Sync()
{
  //TODO
  abort();
}

} // namespace a1
