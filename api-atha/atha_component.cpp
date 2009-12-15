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
#include <iostream>
#include "atha_init.h"
#include "atha_component.h"
#include "atha_timer.h"
#include "atha_debug.h"
#include <queue>
#include <list>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#if defined(KAAPI_USE_IPHONEOS)
#include "KaapiIPhoneInit.h"
#endif

//#define KAAPI_VERBOSE

namespace atha {

// --------------------------------------------------------------------
Properties KaapiComponentManager::prop;

// --------------------------------------------------------------------
Parser KaapiComponentManager::parser;


// --------------------------------------------------------------------
KaapiComponent::KaapiComponent(const std::string& name, int p)
 : _name(name), _refcount(1), _priority(p), _isregistered(false), _is_init(false)
{ }


// --------------------------------------------------------------------
KaapiComponent::~KaapiComponent()
{
}


// --------------------------------------------------------------------
int KaapiComponent::initialize() throw()
{
  if (_is_init) return 0; _is_init = true;
  _startup_time = WallTimer::gettime();
  return 0;
}


// --------------------------------------------------------------------
int KaapiComponent::commit() throw()
{
  return 0;
}


// --------------------------------------------------------------------
void KaapiComponent::is_required_by( KaapiComponent* /*caller*/ ) 
{
  ++_refcount;
  if (KaapiComponentManager::register_object( this )) 
  {
    declare_dependencies();
  }
}


// --------------------------------------------------------------------
void KaapiComponent::add_options( Parser* /*parser*/, Properties* /*global_prop*/ )
{
}


// --------------------------------------------------------------------
struct KaapiComponentRef {
  KaapiComponentRef(KaapiComponent* cr)
   : component(cr)
  { }
  bool operator<( const KaapiComponentRef& cr ) const
  { return component->get_priority() > cr.component->get_priority(); }

  KaapiComponent* component;
};

static std::priority_queue<KaapiComponentRef>* all_components =0;
static std::list<KaapiComponent*>*             all_order_components;


// --------------------------------------------------------------------
bool KaapiComponentManager::register_object( KaapiComponent* obj )
{
  if (obj->is_registered()) return false;
  obj->set_registered();
  //std::cout << "Register Component '" << obj->_name << "' with priority=" << obj->get_priority() << std::endl;
  if (all_components ==0) all_components = new std::priority_queue<KaapiComponentRef>;
  all_components->push( KaapiComponentRef(obj) );
  if (all_order_components ==0) all_order_components = new std::list<KaapiComponent*>;
  return true;
}


// --------------------------------------------------------------------
bool KaapiComponentManager::register_shlib_object( KaapiComponent* obj )
{
  if (obj->is_registered()) return false;
  obj->set_registered();
  if (all_components ==0) all_components = new std::priority_queue<KaapiComponentRef>;
  all_components->push( KaapiComponentRef(obj) );
  if (all_order_components ==0) all_order_components = new std::list<KaapiComponent*>;
  all_order_components->push_back( obj );
  return true;
}


// --------------------------------------------------------------------
int KaapiComponentManager::parseoption(int& argc , char**& argv ) throw()
{
  static bool is_called = false; if (is_called) return 0; is_called = true;

#if defined(KAAPI_USE_APPLE) || defined(KAAPI_USE_IPHONEOS)
  /* seed for rand */
  sranddev();
#elif defined(KAAPI_USE_LINUX)
  srand( getpid() );
#else
#error "No implemented"
#endif
  int err;
  std::list<KaapiComponent*>::iterator ibeg;
  std::list<KaapiComponent*>::iterator iend;
  
  /* Add options to the parser */
  if (all_components !=0)
  {
    while (!all_components->empty()) 
    {
      const KaapiComponentRef& kr = all_components->top();
      all_order_components->push_back( kr.component );
      kr.component->add_options(&KaapiComponentManager::parser, &KaapiComponentManager::prop);
      //std::cout << "Declared Component '" << kr.component->_name << "', priority=" << kr.component->get_priority() << std::endl;
      all_components->pop();
    }
  }

  /* first initialize kaapi_prop from file $HOME/kaapi.rc */
  std::string filename;
  char* name = getenv("HOME");
  if (name !=0) 
  {
    try {
      filename = name;
      filename = filename + "/kaapi.rc";
      KaapiComponentManager::prop.load( filename );
    } catch (const IOError& e) { 
    }
  }

  /* second: initialize kaapi_prop from file $PWD/kaapi.rc */
  name = getcwd( 0, 0 );
  if (name !=0) {
    try {
      filename = name;
      filename = filename + "/kaapi.rc";
      KaapiComponentManager::prop.load( filename );
    } catch (const IOError& e) { 
    }
    free( name );
  }
  
  /* Dump the contents of the command line: */
  std::string cmdline;
  for (int i=0; i<argc; ++i)
    cmdline = cmdline + " " + argv[i];

  /* Parse command line, file and environment */
  KaapiComponentManager::parser.parse( KaapiComponentManager::prop, argc, argv );

  KaapiComponentManager::prop["program.name"] = (argv ==0 ? "no_name" : (argv[0]));
  KaapiComponentManager::prop["program.pwd"] = (getenv("PWD") == 0 ? "." : getenv("PWD"));
  KaapiComponentManager::prop["program.cmdline"] = cmdline;

#if !defined(KAAPI_USE_IPHONEOS) /* cannot do that on iPhone */
  /* recreate program.rootdir entry if it does not exist */
  std::string rootdir = KaapiComponentManager::prop["util.rootdir"];
  if (rootdir == "")
    rootdir = KaapiComponentManager::prop["nameserver.rootdir"];
  if (rootdir == "")
    rootdir = KaapiComponentManager::prop["ft.pwd"];
  
  if (rootdir != "")
  {
    err = mkdir(rootdir.c_str(), S_IRUSR|S_IWUSR|S_IXUSR);
    if ((err != 0) && (errno !=EEXIST))
    {
      logfile() << "[KaapiComponentManager::initialize] cannot create directory: " << rootdir << std::endl;
      perror(0);
      Exception_throw( PosixError(rootdir,errno) );
    }
    /* verify that the path is a directory */
    struct stat stat_pwd;
    err = lstat( rootdir.c_str(), &stat_pwd );
    if ((err !=0) || ( (stat_pwd.st_mode & S_IFDIR) ==0))
    {
      logfile() << "[KaapiComponentManager::initialize] not a directory: '" << rootdir << "'" << std::endl;
      perror(0);
      Exception_throw( PosixError(rootdir,errno) );
    }
    KaapiComponentManager::prop["util.rootdir"] = rootdir;  
  }
#endif  
  return 0;
}
      
// --------------------------------------------------------------------
int KaapiComponentManager::initialize(int& argc, char**& argv) throw()
{
  static bool is_called = false; if (is_called) return 0; is_called = true;
  if (argv ==0) return 0;
  
  KaapiComponentManager::parseoption( argc, argv );

  int err;
  std::list<KaapiComponent*>::iterator ibeg;
  std::list<KaapiComponent*>::iterator iend;
  
  /* Initialize each component */
  if (all_order_components !=0)
  {
    ibeg = all_order_components->begin();
    iend = all_order_components->end();
    while (ibeg != iend) 
    {
      KaapiComponent* comp = *ibeg;  
      //std::cout << "Initialize Component '" << comp->_name << "', priority=" << comp->get_priority() << std::endl;
      if ((err = comp->initialize()) !=0) return err;
      ++ibeg;
    }

    /* Commit each component */
    ibeg = all_order_components->begin();
    iend = all_order_components->end();
    while (ibeg != iend) 
    {
      //std::cout << "Commit Component '" << (*ibeg)->_name << "', priority=" << (*ibeg)->get_priority() << std::endl;
      KaapiComponent* comp = *ibeg;  
      //std::cout << "Commit Component '" << comp->_name << "', priority=" << comp->get_priority() << std::endl;
      if ((err = comp->commit()) !=0) return err;
      ++ibeg;
    }
  }
  
  return 0;
}


// --------------------------------------------------------------------
int KaapiComponentManager::terminate() throw()
{
#if defined(KAAPI_USE_IPHONEOS)
  std::ostringstream buffer;
  buffer << "End of application. Elapse time:" << WallTimer::gettime() - Init::component.startup_time() << std::endl;
  printText(buffer.str().c_str());
  terminateiPhoneKaapi();
  /* wait ad vitam */
  while (1) sleep(10000);
#endif  

  static bool is_called = false; if (is_called) return 0; is_called = true;

  /* terminate each component in the reverse order of thier initialization */
  if (all_order_components !=0)
  {
    std::list<KaapiComponent*>::reverse_iterator ibeg = all_order_components->rbegin();
    std::list<KaapiComponent*>::reverse_iterator iend = all_order_components->rend();
    while (ibeg != iend) 
    {
      if (*ibeg !=0) { 
        KaapiComponent* comp = *ibeg;  
        //std::cout << "Terminate Component '" << comp->_name << "', priority=" << comp->get_priority() << std::endl;
        int err = comp->terminate();
        if (err !=0) return err;
      }
      ++ibeg;
    }
  }
  KaapiComponentManager::prop.clear();
  if (all_components !=0) delete all_components;
  if (all_order_components !=0) delete all_order_components;
  return 0;
}

} // namespace Util
