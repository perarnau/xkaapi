// =========================================================================
// (c) INRIA, projet MOAIS, 2006
// Author: T. Gautier
//
//
//
// =========================================================================
#include "kaapi_impl.h"
#include "kaapi++"
#include "ka_init.h"
#include "ka_error.h"
#include "ka_debug.h"
#include "ka_timer.h"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <signal.h>
#include <limits>
#if not defined(_WIN32)
#include <sys/resource.h> // for core size limit
#endif
#include <sys/types.h>
#include <sys/stat.h>
#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif
#if defined(KAAPI_USE_IPHONEOS)
#include "KaapiIPhoneInit.h"
#endif
#if defined(KAAPI_USE_CPUSET)
#include <cpuset.h>
#endif


namespace ka {

kaapi_bodies_t::kaapi_bodies_t( kaapi_task_body_t cb, kaapi_task_body_t gb )
 : cpu_body(cb), gpu_body( gb ), 
   default_body(cb) /* because we are on a CPU */
{
}


// --------------------------------------------------------------------
Init Init::component;

// --------------------------------------------------------------------
bool Init::verboseon = false;

// --------------------------------------------------------------------
bool Init::enable_trace = false;

// --------------------------------------------------------------------
bool Init::on_term = true;

// --------------------------------------------------------------------
bool Init::on_thread = false;

// --------------------------------------------------------------------
#if defined(KAAPI_DEBUG_MEM)
void* __kaapi_malloc( size_t s)
{ return malloc(s); }
void* __kaapi_calloc( size_t s1, size_t s2)
{ return calloc(s1,s2); }
void __kaapi_free( void* p )
{ return free(p); }
#endif
 
#if 0
// DEAD CODE ?
// --------------------------------------------------------------------
static void kaapi_at_exit()
{
  KaapiComponentManager::terminate();
  KAAPI_ASSERT_M( false, "[kaapi_at_exit] should never be here ????"); 
}
#endif


// --------------------------------------------------------------------
InitKaapiCXX::InitKaapiCXX()
{
  static int iscalled = 0;
  if (iscalled !=0) return;
    
  WrapperFormat<char>::format.reinit(kaapi_char_format);
  WrapperFormat<short>::format.reinit(kaapi_short_format);
  WrapperFormat<int>::format.reinit(kaapi_int_format);
  WrapperFormat<long>::format.reinit(kaapi_long_format);
  WrapperFormat<unsigned char>::format.reinit(kaapi_uchar_format);
  WrapperFormat<unsigned short>::format.reinit(kaapi_ushort_format);
  WrapperFormat<unsigned int>::format.reinit(kaapi_uint_format);
  WrapperFormat<unsigned long>::format.reinit(kaapi_ulong_format);
  WrapperFormat<float>::format.reinit(kaapi_float_format);
  WrapperFormat<double>::format.reinit(kaapi_double_format);
}


// --------------------------------------------------------------------
void Init::set_local_gid( GlobalId newgid )
{
  System::local_gid = newgid;
}


// --------------------------------------------------------------------
Init::Init()
 : KaapiComponent("kaapi", KaapiComponentManager::KERNEL_COMPONENT_PRIORITY)
{}


// --------------------------------------------------------------------
static Parser::Module* kaapi_module = 0;
void Init::add_options( Parser* parser, Properties* global_prop )
{
  if (kaapi_module !=0) return;
  kaapi_module = new Parser::Module;
  kaapi_module->set_name("kaapi");
  kaapi_module->add_option("verboseon", "false", "true iff verbose mode is activated");
  kaapi_module->add_option("thread.cachesize", "0", "number of POSIX cached threads");
  kaapi_module->add_option("thread.stacksize", "65536", "Size of POSIX thread stack");
  kaapi_module->add_option("thread.cpuset", "", "CPU set to use for kernel thread");
  kaapi_module->add_option("coresize", "0", "max size of the coredump when abort (\"0\" for no coredump, \"unlimited\" for unlimited, \"default\" for system value)");
  kaapi_module->add_option("requirednofile", "<no value>", "minimum value required for NOFILE limit");
  kaapi_module->add_option("rootdir", ".", "Directory where to store output file of the application.");
  kaapi_module->add_option("log.type", "thread", "Output log on terminal ('term') or in a file ('file') or in file per thread (thread)");

  kaapi_module->add_option("globalid", "", "Global identifier of the node. A gid is an integer value.");

  //\TODO: no trace in this version
  kaapi_module->add_option("trace.decode", "", "Decode the trace file with name given as parameter");
  kaapi_module->add_option("trace.enable", "false", "Enable or disable trace. Default is no.");
  kaapi_module->add_option("trace.mask", "", "Mask some events, e.g '2,* 3,1 3,0 0,*' mask all events of levels 2 and 0 and mask events 0 and 1 of level 3.", 'a', " ");
  kaapi_module->add_option("trace.umask", "", "Unmask some events, e.g '2,* 3,1 3,0 0,*' unmask all events of levels 2 and 0 and unmask events 0 and 1 of level 3.", 'a', " ");
  kaapi_module->add_option("trace.buffer", "1024", "Maximum buffer size in KBytes. Default is 1MBytes.");

  //\TODO: no trace in this version
  kaapi_module->add_option("ps.enable", "false", "Enable/disable trace, disable for Kaapi application (KaapiServer, KaapiAdmin,...)");
  kaapi_module->add_option("ps.period", "0", "frequency of updatating status of the process (ms)");
  kaapi_module->add_option("ps.format", "all", "format to display statistics. 'all': all statistics, 'total': not average, 'cpu': about cpu, 'ws': about work stealing");
  parser->add_module( kaapi_module, global_prop );
}


// --------------------------------------------------------------------
void Init::declare_dependencies()
{
  /* no dependencies */
}

// --------------------------------------------------------------------
int Init::initialize() throw()
{
  KaapiComponent::initialize();
  static bool is_called = false; if (is_called) return 0; is_called = true;

  Init::verboseon = false;
    
  try {
    /* verboseon is true ? */    
    if (KaapiComponentManager::prop["util.verboseon"] == "true")
    {
      verboseon = true;
    }
    
    /* Core dump */
#if not defined  (_WIN32)
    if (KaapiComponentManager::prop["util.coresize"] != "default")
    {
      struct rlimit core_limit;
      if (getrlimit(RLIMIT_CORE,&core_limit) == 0)
      {
        if (KaapiComponentManager::prop["util.coresize"] == "unlimited")
          core_limit.rlim_cur = RLIM_INFINITY;
        else 
          core_limit.rlim_cur = Parser::String2ULong(KaapiComponentManager::prop["util.coresize"]);
        KAAPI_CPPLOG((setrlimit(RLIMIT_CORE,&core_limit) != 0),"[initialize]: Can't set core file size limit (hard limit too low ?)");
      }
    }  
    
    // check the NOFILE limit
    if (KaapiComponentManager::prop["util.requirednofile"] != "<no value>")
    {
      struct rlimit nofile_limit;
      unsigned long required_limit = Parser::String2ULong(KaapiComponentManager::prop["util.requirednofile"]);
      KAAPI_ASSERT_M(getrlimit(RLIMIT_NOFILE,&nofile_limit) == 0, "[initialize]: Can't get RLIMIT_NOFILE limit");
      KAAPI_ASSERT_M(nofile_limit.rlim_max >= required_limit, "[initialize]: RLIMIT_NOFILE limit too low"); 
      if (nofile_limit.rlim_cur != nofile_limit.rlim_max) {
        nofile_limit.rlim_cur = nofile_limit.rlim_max;
        KAAPI_ASSERT_M(setrlimit(RLIMIT_NOFILE,&nofile_limit) == 0, "[initialize]: Can't set RLIMIT_NOFILE limit");
      }
    }
#endif

    /* Logfile */
    initialize_logfile();

    /* Calibrate clock: after lock_logfile (if logging was used) */
    HighResTimer::calibrate();
  
    /* set the cpu id if mask is defined by environment */
    KaapiComponentManager::prop["util.thread.cpuset"] = getenv("KAAPI_CPUSET");
    
    /* Global Id */
    if (KaapiComponentManager::prop["util.globalid"] == "") 
    {
      System::local_gid = 0;
    }
    else if (KaapiComponentManager::prop["util.globalid"] == "auto")
    {
      std::ostringstream gidtxt;
      gidtxt << getpid() << HighResTimer::gettime();
      System::local_gid = kaapi_hash_value( gidtxt.str().c_str() );
    }
    else
    {
      System::local_gid = (uint32_t)Parser::String2ULong(KaapiComponentManager::prop["util.globalid"]);
    }
    
    /* Set the global id of the process */
    KaapiComponentManager::prop["util.globalid"] = Parser::ULong2String( System::local_gid );
    

    /* \TODO */
    //Format::init_format();

    /* log file */    
    KAAPI_ASSERT_M(  (KaapiComponentManager::prop["util.log.type"] == "term")
                  || (KaapiComponentManager::prop["util.log.type"] == "file")
                  || (KaapiComponentManager::prop["util.log.type"] == "thread"), 
              "[Init::initialize] bad util.log.type option" );
    Init::on_term = KaapiComponentManager::prop["util.log.type"] == "term";
    Init::on_thread = KaapiComponentManager::prop["util.log.type"] == "thread";
    
  } catch (const std::exception& e ) {
    logfile() << "[Init::initialize] catch '" << e.what() << "'" << std::endl;
    return -1;
  } catch ( ... ) {
    logfile() << "[Init::initialize] catch unknown exception" << std::endl;
    return -1;
  }
   
  return 0;
}


// --------------------------------------------------------------------
int Init::terminate() throw()
{
  //logfile() << "In Init::terminate()" << std::endl;
  static bool is_called = false; if (is_called) return 0; is_called = true;

  if ( kaapi_module != 0 ) delete kaapi_module ;

  KAAPI_CPPLOG( Init::verboseon, "[Init::terminate] SignalThread exit");

  return 0;
}

}
