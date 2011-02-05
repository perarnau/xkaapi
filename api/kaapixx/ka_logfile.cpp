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
#include "ka_error.h"
#include "ka_debug.h"
#include "ka_init.h"
#include "ka_timer.h"
#include <map>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>


#include <sys/types.h>
#include <sys/stat.h>
#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif
#if defined(KAAPI_USE_IPHONEOS)
#include "KaapiIPhoneInit.h"
#include <streambuf>
#endif


namespace ka {

// --------------------------------------------------------------------
void initialize_logfile()
{
}

// --------------------------------------------------------------------
void lock_logfile()
{
}

// --------------------------------------------------------------------
void unlock_logfile()
{
}


#if defined(KAAPI_USE_IPHONEOS)
/* stream buf to redirect ostream into buffer */
class logoutbuf : public std::basic_streambuf<char, std::char_traits<char> > {
private:
  typedef std::char_traits<char> traits;
  typedef std::basic_streambuf<char,traits>::int_type int_type;
  typedef std::vector<char> buffer_type;
  buffer_type _buffer;

  virtual int_type overflow(int_type c) {
    if(!traits::eq_int_type(c, traits::eof())) {
      _buffer.push_back(c);
    }
    return traits::not_eof(c);
  }
  virtual std::streamsize xsputn(const char* s, std::streamsize num) 
  {
    std::copy(s, s + num, std::back_inserter<buffer_type>(_buffer));
    return num;
  }
  virtual int sync() 
  {
    if(!_buffer.empty()) 
    {
      _buffer.push_back(0);
      printText( &_buffer[0] );
      _buffer.clear();
    }
    return 0;
  }
};

class logoutbuf_init {
private:
  typedef std::char_traits<char> traits;
  logoutbuf _buf;

public:
  logoutbuf* buf() {
    return &_buf;
  }
};


class logostream : private virtual logoutbuf_init,
                   public std::basic_ostream<char, std::char_traits<char> > {
private:
  typedef std::char_traits<char> traits;
  typedef logoutbuf_init logoutbuf_init;
public:
  logostream()
      : logoutbuf_init(),
        std::basic_ostream<char,traits>(logoutbuf_init::buf()) 
  {
  }
};
static logostream iphone_out;
#endif


// --------------------------------------------------------------------
static std::ofstream* fout_per_process = 0;
static std::map<kaapi_processor_id_t,std::ofstream*> fout_per_kprocessor;

std::ostream& logfile()
{
#if defined(KAAPI_USE_IPHONEOS)
  return iphone_out;
#else
  kaapi_processor_t* kproc = kaapi_get_current_processor();
  int self = (kproc == 0 ? -1 : kproc->kid);
  if (Init::on_thread) 
  {
    std::map<kaapi_processor_id_t,std::ofstream*>::iterator curr = fout_per_kprocessor.find(self);
    if (curr == fout_per_kprocessor.end())
    {
      /* create the directory & file for output */
      std::string pwdlog = ka::KaapiComponentManager::prop["util.rootdir"] + "/log";
#if defined (_WIN32)
      int err = mkdir(pwdlog.c_str());
#else
      int err = mkdir(pwdlog.c_str(), S_IRWXU | S_IRGRP | S_IXGRP);
#endif
      if ((err != 0) && (errno !=EEXIST))
      {
        std::ostringstream msg;
        msg << "[ka::Statistics::initialize] cannot create directory: " << pwdlog << std::endl;
        throw std::runtime_error(msg.str()); //,errno) );
      }
      std::ostringstream sname;
      sname << pwdlog << "/cout." << ka::System::local_gid << "." << self << "." << getpid();
      std::ofstream* fout = new std::ofstream(sname.str().c_str());
      KAAPI_ASSERT_M( fout->good(), "cannot create per process cout file");
      fout_per_kprocessor.insert( std::make_pair(self, fout) );
      unlock_logfile();
      return *fout;
    }
    return *(curr->second);
  }
  if (Init::on_term) 
  {
    std::cout << System::local_gid << "::" << self << "::[" << std::setw(9) << std::setprecision(7) << std::showpoint; 
    std::cout.fill( '0' );
    std::cout.setf( std::ios_base::left, std::ios_base::adjustfield);
    std::cout << double(kaapi_get_elapsedns() - kaapi_default_param.startuptime)*1e-6 << "]: ";
    return std::cout;
  }
  if (fout_per_process ==0) 
  { 
    std::string pwdlog = ka::KaapiComponentManager::prop["util.rootdir"] + "/log";
#if defined (_WIN32)
    int err = mkdir(pwdlog.c_str());
#else
    int err = mkdir(pwdlog.c_str(), S_IRWXU | S_IRGRP | S_IXGRP);
#endif
    if ((err != 0) && (errno !=EEXIST))
    {
      std::ostringstream msg;
      msg << "[ka::Statistics::initialize] cannot create directory: " << pwdlog << std::endl;
      throw std::runtime_error(msg.str()); //,errno) );
    }
    std::ostringstream sname; 
    sname << pwdlog << "/cout." << ka::System::local_gid << "." << getpid();
    fout_per_process = new std::ofstream(sname.str().c_str());
    KAAPI_ASSERT_M( fout_per_process->good(), "cannot create per process cout file");
  }
  *fout_per_process << std::flush << ka::System::local_gid << "::" << self << "::[" << std::setw(9) << std::setprecision(7); 
  fout_per_process->fill( '0' );
  fout_per_process->setf( std::ios_base::left, std::ios_base::adjustfield);
  *fout_per_process << ka::WallTimer::gettime() - Init::component.startup_time() << "]: ";
  return *fout_per_process;
#endif
}


}
