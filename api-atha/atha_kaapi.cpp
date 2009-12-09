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
Format::Format( 
        const std::string& name,
        size_t             size,
        void             (*cstor)( void* dest),
        void             (*dstor)( void* dest),
        void             (*cstorcopy)( void* dest, const void* src),
        void             (*copy)( void* dest, const void* src),
        void             (*assign)( void* dest, const void* src),
        void             (*print)( FILE* file, const void* src)
)
{
  kaapi_format_register( this, strdup(name.c_str()));
  this->size      = size;
  this->cstor     = cstor;
  this->dstor     = dstor;
  this->cstorcopy = cstorcopy;
  this->copy      = copy;
  this->assign    = assign;
  this->print     = print;
}


// --------------------------------------------------------------------------
FormatUpdateFnc::FormatUpdateFnc( 
  const std::string& name,
  int (*update_mb)(void* data, const struct kaapi_format_t* fmtdata,
                   const void* value, const struct kaapi_format_t* fmtvalue )
) : Format::Format(name, 0, 0, 0, 0, 0, 0, 0)
{
  this->update_mb = update_mb;
}

// --------------------------------------------------------------------------
template <>
const Format* WrapperFormat<kaapi_int8_t>::format = (const Format*)&kaapi_char_format;
template <>
const Format* WrapperFormat<kaapi_int16_t>::format = (const Format*)&kaapi_short_format;
template <>
const Format* WrapperFormat<kaapi_int32_t>::format = (const Format*)&kaapi_int_format;
/* TODO: switch vers format_longlong si int64_t == long long */
template <>
const Format* WrapperFormat<kaapi_int64_t>::format = (const Format*)&kaapi_long_format;
template <>
const Format* WrapperFormat<kaapi_uint8_t>::format = (const Format*)&kaapi_uchar_format;
template <>
const Format* WrapperFormat<kaapi_uint16_t>::format = (const Format*)&kaapi_ushort_format;
template <>
const Format* WrapperFormat<kaapi_uint32_t>::format = (const Format*)&kaapi_uint_format;
/* TODO: switch vers format_longlong si int64_t == long long */
template <>
const Format* WrapperFormat<kaapi_uint64_t>::format = (const Format*)&kaapi_ulong_format;
template <>
const Format* WrapperFormat<float>::format = (const Format*)&kaapi_float_format;
template <>
const Format* WrapperFormat<double>::format = (const Format*)&kaapi_double_format;

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
  Sync();
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
  int err;
  kaapi_stack_t* stack = kaapi_self_stack();
redo:
  err = kaapi_stack_execall(stack);
  if (err == EWOULDBLOCK)
  {
    kaapi_sched_suspend( kaapi_get_current_processor() );
    goto redo;
  }
}


// --------------------------------------------------------------------
void _athakaapi_dummy(void*)
{
}

Community* zcom =0;

// --------------------------------------------------------------------
void __attribute__ ((constructor)) atha_init()
{
  int argc = 1;
  char**argv = 0;
//  zcom = new a1::Community(a1::System::join_community( argc, argv ));
}

// --------------------------------------------------------------------
void __attribute__ ((destructor)) atha_fini()
{
// TG: TODO: order between this call and kaapi_fini !!!  zcom->leave();
}


} // namespace a1
