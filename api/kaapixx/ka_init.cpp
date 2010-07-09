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
//TODO with network: #include "atha_format.h"
//TODO #include "utils_signal.h"
//TODO #include "utils_trace_buffer.h"
//TODO #include "utils_trace_recorder.h"
#include <iostream>
#include <iomanip>
#include <signal.h>
#include <limits>
#if not defined(_WIN32)
#include <sys/resource.h> // for core size limit
#endif
#include <sys/types.h>
#include <sys/stat.h>
#if defined(KAAPI_USE_DARWIN)
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
kaapi_uint32_t Init::local_gid = kaapi_uint32_t( -1U ); // <-> GlobalId::NO_SITE;

// --------------------------------------------------------------------
Architecture Init::local_archi;

// --------------------------------------------------------------------
bool Init::enable_trace = false;

// --------------------------------------------------------------------
bool Init::on_term = true;

// --------------------------------------------------------------------
bool Init::on_thread = false;

// --------------------------------------------------------------------
void* __kaapi_malloc( size_t s)
{ return malloc(s); }
void* __kaapi_calloc( size_t s1, size_t s2)
{ return calloc(s1,s2); }
void __kaapi_free( void* p )
{ return free(p); }
 
// --------------------------------------------------------------------
void kaapi_at_exit()
{
  KaapiComponentManager::terminate();
  KAAPI_ASSERT_M( false, "[kaapi_at_exit] should never be here ????"); 
}



// --------------------------------------------------------------------
InitKaapiCXX::InitKaapiCXX()
{
  static int iscalled = 0;
  if (iscalled !=0) return;
  
  kaapi_init();
  
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
    KaapiComponentManager::prop["util.thread.cpuset"] = getenv("KAAPI_CPU_SET");
    
    /* Global Id */
    if (KaapiComponentManager::prop["util.globalid"] == "") 
    {
      Init::local_gid = 0;
    }
    else if (KaapiComponentManager::prop["util.globalid"] == "auto")
    {
      std::ostringstream gidtxt;
      gidtxt << getpid() << HighResTimer::gettime();
      Init::local_gid = kaapi_hash_value( gidtxt.str().c_str() );
    }
    else
    {
      Init::local_gid = Parser::String2ULong(KaapiComponentManager::prop["util.globalid"]);
    }
    
    /* Set the global id of the process */
    KaapiComponentManager::prop["util.globalid"] = Parser::ULong2String( Init::local_gid );
    

    /* Compute local archi */
    Init::local_archi.set_local();

    /* \TODO */
    //Format::init_format();

    /* log file */    
    KAAPI_ASSERT_M( (KaapiComponentManager::prop["util.log.type"] == "term")
                  || (KaapiComponentManager::prop["util.log.type"] == "file")
                  || (KaapiComponentManager::prop["util.log.type"] == "thread"), 
              "[Init::initialize] bad util.log.type option" );
    Init::on_term = KaapiComponentManager::prop["util.log.type"] == "term";
    Init::on_thread = KaapiComponentManager::prop["util.log.type"] == "thread";
    
  } catch (const Exception& e ) {
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


// -------------------------------------------------------------------------
void Architecture::set_local()
{
  kaapi_uint8_t endian, sz_l, sz_ld, sz_b;
  #ifdef WORDS_BIGENDIAN
    endian = 0x01;
  #else
    endian = 0x00;
  #endif
  if (sizeof(long) == 8) {
    sz_l = 0x02;
  } else {
    sz_l = 0x00;
  }
  if (sizeof(bool) == 4) {
    sz_b = 0x04;
  } else {
    sz_b = 0x00;
  }
  sz_ld = (LongDouble::get_local_format() << 3);
  _archi = sz_b | sz_ld | sz_l | endian;
}

// -------------------------------------------------------------------------
// --------- Conversion of long double -------------------------------------

#define DEBUG_GET false
#define DEBUG_SET false

struct FormatInfo {
  exptype exp_special;
  exptype exp_bias;
  unsigned int mant_size;
  size_t size;
};

                             //  exp_special, exp_bias, mant_size, size 
FormatInfo infos[NB_FORMAT] = { 
                                {    0x7ff  ,  1023   ,     52   ,  8  }, // IEEE_DOUBLE
                                {   0x7fff  , 16383   ,     63   , 12  }, // IEEE_EXTENDED_12
                                {   0x7fff  , 16383   ,     63   , 16  }, // IEEE_EXTENDED_16
                                {   0x7fff  , 16383   ,    112   , 16  }, // IEEE_QUADRUPLE
                                {    0x7ff  ,  1023   ,     52   , 16  }, // PPC_QUADWORD (coded as a double, the second double is null)
                              };

// ----------------------------------------------------------

void Sign::get(LongDoubleFormat f, unsigned char *r){
  switch(f){
    case IEEE_DOUBLE:
      _positive = ((r[7] & 0x80) == 0x00);
      break;
    case IEEE_EXTENDED_12:
    case IEEE_EXTENDED_16:
      _positive = ((r[9] & 0x80) == 0x00);
      break;         
    case IEEE_QUADRUPLE:
      _positive = ((r[15] & 0x80) == 0x00);
      break;  
    case PPC_QUADWORD:
      _positive = ((r[15] & 0x80) == 0x00);
      break;
    default:
      KAAPI_ASSERT_M( false , "Unknown LongDoubleFormat");
      break;
  }
}


void Sign::set(LongDoubleFormat f, unsigned char *r){
  switch(f){
    case IEEE_DOUBLE:
      r[7] = ((r[7] & 0x7f) | (_positive ? 0x00 : 0x80));
      break;
    case IEEE_EXTENDED_12:
    case IEEE_EXTENDED_16:
      r[9] = ((r[9] & 0x7f) | (_positive ? 0x00 : 0x80));
      break;         
    case IEEE_QUADRUPLE:
      r[15] = ((r[15] & 0x7f) | (_positive ? 0x00 : 0x80));
      break;  
    case PPC_QUADWORD:
      r[15] = ((r[15] & 0x7f) | (_positive ? 0x00 : 0x80));
      break;
    default:
      KAAPI_ASSERT_M( false , "Unknown LongDoubleFormat");
      break;
  }
}


// ----------------------------------------------------------


void Exponent::get(LongDoubleFormat f, unsigned char *r){
  switch(f){
    case IEEE_DOUBLE:
      _exp = ((r[7] & 0x7f) << 4) | ((r[6] & 0xf0) >> 4);
      break;
    case IEEE_EXTENDED_12:
    case IEEE_EXTENDED_16:
      _exp = ((r[9] & 0x7f) << 8) | r[8] ;
      break;
    case IEEE_QUADRUPLE:
      _exp = ((r[15] & 0x7f) << 8) | r[14] ;    
      break;
    case PPC_QUADWORD:
      _exp = ((r[15] & 0x7f) << 4) | ((r[14] & 0xf0) >> 4);
      break;
    default:
      KAAPI_ASSERT_M( false , "Unknown LongDoubleFormat");
      break;
  }
}

void Exponent::set(LongDoubleFormat f, unsigned char *r){
  switch(f){
    case IEEE_DOUBLE:
      r[6] = (r[6] & 0x0f) | ((_exp & 0x000f) << 4);
      r[7] = (r[7] & 0x80) | ((_exp & 0x07f0) >> 4);
      break;
    case IEEE_EXTENDED_12:
    case IEEE_EXTENDED_16:
      r[7] = (r[7] & 0x7f) | (_exp == 0 ? 0x00 : 0x80); // explicit bit for normalization
      r[8] = (_exp & 0x00ff);
      r[9] = (r[9] & 0x80) | ((_exp & 0x7f00) >> 8);  
      break;
    case IEEE_QUADRUPLE:
      r[14] = (_exp & 0x00ff);
      r[15] = (r[15] & 0x80) | ((_exp & 0x7f00) >> 8);    
      break;
    case PPC_QUADWORD:
      r[14] = (r[14] & 0x0f) | ((_exp & 0x000f) << 4);
      r[15] = (r[15] & 0x80) | ((_exp & 0x07f0) >> 4);
      break;
    default:
      KAAPI_ASSERT_M( false , "Unknown LongDoubleFormat");
      break;
  }
}

// ----------------------------------------------------------

void Mantissa::get(LongDoubleFormat f, unsigned char *r){
  int i;
  switch(f){
    case IEEE_DOUBLE:
      i = MANTISSA_MAXSIZE;
      _mant[--i] = ((r[6] & 0x0f) << 4) | ((r[5] & 0xf0) >> 4);
      _mant[--i] = ((r[5] & 0x0f) << 4) | ((r[4] & 0xf0) >> 4);
      _mant[--i] = ((r[4] & 0x0f) << 4) | ((r[3] & 0xf0) >> 4);
      _mant[--i] = ((r[3] & 0x0f) << 4) | ((r[2] & 0xf0) >> 4);
      _mant[--i] = ((r[2] & 0x0f) << 4) | ((r[1] & 0xf0) >> 4);
      _mant[--i] = ((r[1] & 0x0f) << 4) | ((r[0] & 0xf0) >> 4);
      _mant[--i] = ((r[0] & 0x0f) << 4);
      for ( ; i > 0 ; ){
        _mant[--i] = 0x00;      
      }
      break;
    case IEEE_EXTENDED_12:
    case IEEE_EXTENDED_16:
      i = MANTISSA_MAXSIZE;
      _mant[--i] = ((r[7] & 0x7f) << 1) | ((r[6] & 0x80) >> 7);
      _mant[--i] = ((r[6] & 0x7f) << 1) | ((r[5] & 0x80) >> 7);
      _mant[--i] = ((r[5] & 0x7f) << 1) | ((r[4] & 0x80) >> 7);
      _mant[--i] = ((r[4] & 0x7f) << 1) | ((r[3] & 0x80) >> 7);
      _mant[--i] = ((r[3] & 0x7f) << 1) | ((r[2] & 0x80) >> 7);
      _mant[--i] = ((r[2] & 0x7f) << 1) | ((r[1] & 0x80) >> 7);
      _mant[--i] = ((r[1] & 0x7f) << 1) | ((r[0] & 0x80) >> 7);
      _mant[--i] = ((r[0] & 0x7f) << 1);
      while(i > 0){
        _mant[--i] = 0x00;      
      }
      break;
    case IEEE_QUADRUPLE:
      i = MANTISSA_MAXSIZE;
      _mant[--i] = r[13];
      _mant[--i] = r[12];
      _mant[--i] = r[11];
      _mant[--i] = r[10];
      _mant[--i] = r[9];
      _mant[--i] = r[8];
      _mant[--i] = r[7];
      _mant[--i] = r[6];
      _mant[--i] = r[5];
      _mant[--i] = r[4];
      _mant[--i] = r[3];
      _mant[--i] = r[2];
      _mant[--i] = r[1];
      _mant[--i] = r[0];
      for ( ; i > 0 ; ){
        _mant[--i] = 0x00;      
      }      
      break;
    case PPC_QUADWORD:
      i = MANTISSA_MAXSIZE;
      _mant[--i] = ((r[14] & 0x0f) << 4) | ((r[13] & 0xf0) >> 4);
      _mant[--i] = ((r[13] & 0x0f) << 4) | ((r[12] & 0xf0) >> 4);
      _mant[--i] = ((r[12] & 0x0f) << 4) | ((r[11] & 0xf0) >> 4);
      _mant[--i] = ((r[11] & 0x0f) << 4) | ((r[10] & 0xf0) >> 4);
      _mant[--i] = ((r[10] & 0x0f) << 4) | ((r[9] & 0xf0) >> 4);
      _mant[--i] = ((r[9] & 0x0f) << 4) | ((r[8] & 0xf0) >> 4);
      _mant[--i] = ((r[8] & 0x0f) << 4);
      for ( ; i > 0 ; ){
        _mant[--i] = 0x00;      
      }
      break;      
    default:
      KAAPI_ASSERT_M( false , "Unknown LongDoubleFormat");
      break;
  }
}


void Mantissa::set(LongDoubleFormat f, unsigned char *r){
  switch(f){
    case IEEE_DOUBLE:
      r[6] = (r[6] & 0xf0) | ((_mant[MANTISSA_MAXSIZE-1] & 0xf0) >> 4);
      r[5] = ((_mant[MANTISSA_MAXSIZE-1] & 0x0f) << 4) | ((_mant[MANTISSA_MAXSIZE-2] & 0xf0) >> 4);
      r[4] = ((_mant[MANTISSA_MAXSIZE-2] & 0x0f) << 4) | ((_mant[MANTISSA_MAXSIZE-3] & 0xf0) >> 4);
      r[3] = ((_mant[MANTISSA_MAXSIZE-3] & 0x0f) << 4) | ((_mant[MANTISSA_MAXSIZE-4] & 0xf0) >> 4);
      r[2] = ((_mant[MANTISSA_MAXSIZE-4] & 0x0f) << 4) | ((_mant[MANTISSA_MAXSIZE-5] & 0xf0) >> 4);
      r[1] = ((_mant[MANTISSA_MAXSIZE-5] & 0x0f) << 4) | ((_mant[MANTISSA_MAXSIZE-6] & 0xf0) >> 4);
      r[0] = ((_mant[MANTISSA_MAXSIZE-6] & 0x0f) << 4) | ((_mant[MANTISSA_MAXSIZE-7] & 0xf0) >> 4);
      break;
    case IEEE_EXTENDED_12:
    case IEEE_EXTENDED_16:
      r[7] = (r[7] & 0x80) | ((_mant[MANTISSA_MAXSIZE-1] & 0xfe) >> 1);
      r[6] = ((_mant[MANTISSA_MAXSIZE-1] & 0x01) << 7) | ((_mant[MANTISSA_MAXSIZE-2] & 0xfe) >> 1);
      r[5] = ((_mant[MANTISSA_MAXSIZE-2] & 0x01) << 7) | ((_mant[MANTISSA_MAXSIZE-3] & 0xfe) >> 1);
      r[4] = ((_mant[MANTISSA_MAXSIZE-3] & 0x01) << 7) | ((_mant[MANTISSA_MAXSIZE-4] & 0xfe) >> 1);
      r[3] = ((_mant[MANTISSA_MAXSIZE-4] & 0x01) << 7) | ((_mant[MANTISSA_MAXSIZE-5] & 0xfe) >> 1);
      r[2] = ((_mant[MANTISSA_MAXSIZE-5] & 0x01) << 7) | ((_mant[MANTISSA_MAXSIZE-6] & 0xfe) >> 1);
      r[1] = ((_mant[MANTISSA_MAXSIZE-6] & 0x01) << 7) | ((_mant[MANTISSA_MAXSIZE-7] & 0xfe) >> 1);
      r[0] = ((_mant[MANTISSA_MAXSIZE-7] & 0x01) << 7) | ((_mant[MANTISSA_MAXSIZE-8] & 0xfe) >> 1);
      break;
    case IEEE_QUADRUPLE:
      r[13] = _mant[MANTISSA_MAXSIZE-1];
      r[12] = _mant[MANTISSA_MAXSIZE-2];
      r[11] = _mant[MANTISSA_MAXSIZE-3];
      r[10] = _mant[MANTISSA_MAXSIZE-4];
      r[9]  = _mant[MANTISSA_MAXSIZE-5];
      r[8]  = _mant[MANTISSA_MAXSIZE-6];
      r[7]  = _mant[MANTISSA_MAXSIZE-7];
      r[6]  = _mant[MANTISSA_MAXSIZE-8];
      r[5]  = _mant[MANTISSA_MAXSIZE-9];
      r[4]  = _mant[MANTISSA_MAXSIZE-10];
      r[3]  = _mant[MANTISSA_MAXSIZE-11];
      r[2]  = _mant[MANTISSA_MAXSIZE-12];
      r[1]  = _mant[MANTISSA_MAXSIZE-13];
      r[0]  = _mant[MANTISSA_MAXSIZE-14];
      break;
    case PPC_QUADWORD:
      r[14] = (r[14] & 0xf0) | ((_mant[MANTISSA_MAXSIZE-1] & 0xf0) >> 4);
      r[13] = ((_mant[MANTISSA_MAXSIZE-1] & 0x0f) << 4) | ((_mant[MANTISSA_MAXSIZE-2] & 0xf0) >> 4);
      r[12] = ((_mant[MANTISSA_MAXSIZE-2] & 0x0f) << 4) | ((_mant[MANTISSA_MAXSIZE-3] & 0xf0) >> 4);
      r[11] = ((_mant[MANTISSA_MAXSIZE-3] & 0x0f) << 4) | ((_mant[MANTISSA_MAXSIZE-4] & 0xf0) >> 4);
      r[10] = ((_mant[MANTISSA_MAXSIZE-4] & 0x0f) << 4) | ((_mant[MANTISSA_MAXSIZE-5] & 0xf0) >> 4);
      r[9]  = ((_mant[MANTISSA_MAXSIZE-5] & 0x0f) << 4) | ((_mant[MANTISSA_MAXSIZE-6] & 0xf0) >> 4);
      r[8]  = ((_mant[MANTISSA_MAXSIZE-6] & 0x0f) << 4) | ((_mant[MANTISSA_MAXSIZE-7] & 0xf0) >> 4);
      r[7]  = 0;
      r[6]  = 0;
      r[5]  = 0;
      r[4]  = 0;
      r[3]  = 0;
      r[2]  = 0;
      r[1]  = 0;
      r[0]  = 0;
      break;
    default:
      KAAPI_ASSERT_M( false , "Unknown LongDoubleFormat");
      break;
  }
}

void Mantissa::clear(){
  for (int i = 0 ; i < MANTISSA_MAXSIZE ; i++) _mant[i] = 0;
}

void Mantissa::shift_left(){
  for (int i = MANTISSA_MAXSIZE-1 ; i > 0 ; i--) {
    _mant[i] = ((_mant[i] & 0x7f) << 1) | ((_mant[i-1] & 0x80) >> 7);
  }
  _mant[0] = ((_mant[0] & 0x7f) << 1);
}

void Mantissa::shift_right(bool b){
  for (int i = 0 ; i < MANTISSA_MAXSIZE-1 ; i++) {
    _mant[i] = ((_mant[i+1] & 0x01) << 7) | ((_mant[i] & 0xfe) >> 1);
  }
  if (b) {
    _mant[MANTISSA_MAXSIZE-1] = ((_mant[MANTISSA_MAXSIZE-1] & 0xfe) >> 1) | 0x80;
  } else {
    _mant[MANTISSA_MAXSIZE-1] = ((_mant[MANTISSA_MAXSIZE-1] & 0xfe) >> 1);
  }
}

// ----------------------------------------------------------

void LongDouble::get(LongDoubleFormat f, unsigned char *r){

  // Read the values
  _sign.get(f, r);
  _exp.get(f, r);
  _mant.get(f, r);
//  if (DEBUG_GET) debug("test");
  KAAPI_ASSERT_M( _exp._exp >= 0 , "Bad Assertion");

  // Look for special numbers
  if (_exp._exp == infos[f].exp_special) {
    if (DEBUG_GET) printf("-> special number\n");
    _type = LD_INFINITY;
    for (int i = 0 ; i < MANTISSA_MAXSIZE ; i++){
      if (_mant._mant[i] != 0) {
        _type = LD_NAN;
        break;
      }
    }
  } else if (_exp._exp == 0) {
    // if denormalized, we normalize it
    if (DEBUG_GET) printf("-> denormalized\n");
    int d = infos[f].mant_size;
    for ( ; d > 0 ; d--) {
      if ((_mant._mant[MANTISSA_MAXSIZE-1] & 0x80) != 0) break;
      _mant.shift_left();
      _exp._exp--;
    }
    _mant.shift_left();      
    if (d == 0) {
    if (DEBUG_GET) printf("-> zero\n");
      _type = LD_ZERO;
    } else {
    if (DEBUG_GET) printf("-> normalized\n");
      _type = LD_NORMAL;
    }
  } else {
    if (DEBUG_GET) printf("-> normalized\n");
    _type = LD_NORMAL;    
  } 
  
  // Get the real value of the exponent
  _exp._exp -= infos[f].exp_bias;

  if (DEBUG_GET) debug("Fin_du_get");
}

void LongDouble::set(LongDoubleFormat f, unsigned char *r){
  if (DEBUG_SET) debug("Debut_du_set");
  if (_type == LD_INFINITY) {
    if (DEBUG_SET) printf("_type == INFINITY\n");
    _exp._exp = infos[f].exp_special;
    _mant.clear();
  } else if (_type == LD_ZERO) {
    if (DEBUG_SET) printf("_type == ZERO\n");
    _exp._exp = 0;
    _mant.clear();  
  } else if (_type == LD_NAN) {
    if (DEBUG_SET) printf("_type == NAN\n");
    _exp._exp = infos[f].exp_special;
    if ((_mant._mant[MANTISSA_MAXSIZE-1] & 0x80) == 0) {
      // sNaN
      _mant.clear();
      _mant._mant[MANTISSA_MAXSIZE-1] = 0x40;
    } else {
      // qNaN
      _mant.clear();
      _mant._mant[MANTISSA_MAXSIZE-1] = 0x80;
    }
  } else if (_type == LD_NORMAL) {
    _exp._exp += infos[f].exp_bias;
    if (DEBUG_SET) printf("_type == NORMAL -> ");
    if (_exp._exp > 2*infos[f].exp_bias) {
      // Overflow
      if (DEBUG_SET) printf("Overflow\n");
      _exp._exp = infos[f].exp_special;
      _mant.clear();      
    } else if (_exp._exp > 0) {
      // Normalized
      if (DEBUG_SET) printf("Normalized\n");
    } else if (_exp._exp >= (1-(kaapi_int32_t)infos[f].mant_size)) {
      // Denormalized
      if (DEBUG_SET) printf("Denormalized\n");
      _mant.shift_right(true);
      for ( ; _exp._exp < 0 ; _exp._exp++) {
        _mant.shift_right(false);      
      }
    } else {
      // Underflow
      if (DEBUG_SET) printf("Underflow\n");
      _exp._exp = 0;
      _mant.clear();        
    }
  } else {
    KAAPI_ASSERT_M( false , "Internal error");  
  }
  
  KAAPI_ASSERT_M( _exp._exp >= 0 , "Bad Assertion");  
  _sign.set(f, r);  
  _exp.set(f, r);  
  _mant.set(f, r);  
}


LongDoubleFormat LongDouble::local_format = LongDouble::compute_local_format();

LongDoubleFormat LongDouble::compute_local_format()
{
  if (sizeof(long double) == 8){
    return IEEE_DOUBLE;
  } else if (sizeof(long double) == 12) {
    return IEEE_EXTENDED_12;
  } else if (sizeof(long double) == 16) {
    if (std::numeric_limits<long double>::digits == 113){
      return IEEE_QUADRUPLE;
    } else if (std::numeric_limits<long double>::digits == 106){
      return PPC_QUADWORD;
    } else if (std::numeric_limits<long double>::digits == 64){
      return IEEE_EXTENDED_16;
    } else {
      KAAPI_ASSERT_M( false , "Unable too use long double : unknown format");
    }
  } else {
    KAAPI_ASSERT_M( false , "Unable too use long double : unknown format");
  }
}



size_t LongDouble::get_size(LongDoubleFormat f){
  return infos[f].size;
}

void LongDouble::debug(const char *s) {
  printf("%s - _type = %d\n", s, _type);
  printf("%s - _sign = %s\n", s, (_sign._positive ? "true" : "false"));
  printf("%s - _exp  = %d\n", s, _exp._exp);
  printf("%s - _mant = ", s);
  for (int k = MANTISSA_MAXSIZE-1 ; k >= 0 ; k--){
    printf("%02x ",_mant._mant[k]);
  }
  printf("\n");
}

void LongDouble::conversion(LongDoubleFormat from, LongDoubleFormat to, unsigned char *r)
{
  if (from != to) {
    // Try to get more precision when converting PPC_QUADWORD
    // but problem : snan become a qnan
    if ((from == PPC_QUADWORD) && (to != IEEE_DOUBLE)) { 
      KAAPI_ASSERT_M( sizeof(double) == 8, "Error : sizeof(double) is not 8.");
      double *d = (double *)r;
      #ifdef WORDS_BIGENDIAN
        ByteSwap( (unsigned char *)d, sizeof(double));
        ByteSwap( (unsigned char *)(d+1), sizeof(double));
      #endif

      long double* pld = new long double;
      *pld = ((long double) d[0]) + ((long double) d[1]);

      #ifdef WORDS_BIGENDIAN
        ByteSwap( (unsigned char *)pld , sizeof(long double));
      #endif
      LongDouble ld;
      ld.get(get_local_format(), (unsigned char*)pld);
      ld.set(to, r);
      delete pld;
    } else {
      LongDouble ld;
      ld.get(from, r);
      ld.set(to, r);
    }
  }
  // WARN: return value ????
}

void LongDouble::conversion_from_local_to(LongDoubleFormat to, unsigned char *r)
{
  conversion(get_local_format(), to, r);
}

void LongDouble::conversion_to_local_from(LongDoubleFormat from, unsigned char *r)
{
  conversion(from, get_local_format(), r);
}


// --------------------------------------------------------------------
InterfaceAllocator::~InterfaceAllocator() {}


// --------------------------------------------------------------------
InterfaceDeallocator::~InterfaceDeallocator() {}


}
