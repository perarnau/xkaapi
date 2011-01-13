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
  static std::string fmt_name = std::string("__Z4TypeI")+name+"E";
  if (fmt ==0) 
    fmt = new kaapi_format_t;
  kaapi_format_register( fmt, strdup(fmt_name.c_str()));
  fmt->size      = (uint32_t)size;
  fmt->cstor     = cstor;
  fmt->dstor     = dstor;
  fmt->cstorcopy = cstorcopy;
  fmt->copy      = copy;
  fmt->assign    = assign;
  fmt->print     = print;
}


// --------------------------------------------------------------------------
Format::Format( kaapi_format_t* f ) 
 : fmt(f) 
{
}

// --------------------------------------------------------------------------
void Format::reinit( kaapi_format_t* f ) const
{
  fmt = f;
}

// --------------------------------------------------------------------------
struct kaapi_format_t* Format::get_c_format() 
{ 
  if (fmt ==0) fmt = new kaapi_format_t;
  return fmt; 
}

// --------------------------------------------------------------------------
const struct kaapi_format_t* Format::get_c_format() const 
{ 
  if (fmt ==0) fmt = new kaapi_format_t;
  return fmt; 
}

// --------------------------------------------------------------------------
FormatUpdateFnc::FormatUpdateFnc( 
  const std::string& name,
  int (*update_mb)(void* data, const struct kaapi_format_t* fmtdata,
                   const void* value, const struct kaapi_format_t* fmtvalue )
) : Format::Format(name, 0, 0, 0, 0, 0, 0, 0)
{
  fmt->update_mb = update_mb;
}


// --------------------------------------------------------------------------
FormatTask::FormatTask( 
  const std::string&          name,
  size_t                      size,
  int                         count,
  const kaapi_access_mode_t   mode_param[],
  const kaapi_offset_t        offset_param[],
  const kaapi_offset_t        offset_version[],
  const kaapi_format_t*       fmt_param[],
  const size_t                size_param[],
  const kaapi_reducor_t       reducor_param[]
) : Format(0)
{
  if (fmt ==0)
    fmt = new kaapi_format_t;
  kaapi_format_taskregister_static( 
        fmt,
        0, 
        name.c_str(),
        size,
        count,
        mode_param,
        offset_param,
        offset_version,
        fmt_param,
        size_param,
        reducor_param
  );
}


// --------------------------------------------------------------------------
FormatTask::FormatTask( 
  const std::string&          name,
  size_t                      size,
  size_t                    (*get_count_params)(const struct kaapi_format_t*, const void*),
  kaapi_access_mode_t       (*get_mode_param)  (const struct kaapi_format_t*, unsigned int, const void*),
  void*                     (*get_off_param)   (const struct kaapi_format_t*, unsigned int, const void*),
  kaapi_access_t            (*get_access_param)(const struct kaapi_format_t*, unsigned int, const void*),
  void                      (*set_access_param)(const struct kaapi_format_t*, unsigned int, void*, const kaapi_access_t*),
  const kaapi_format_t*     (*get_fmt_param)   (const struct kaapi_format_t*, unsigned int, const void*),
  size_t                    (*get_size_param)  (const struct kaapi_format_t*, unsigned int, const void*),
  void                      (*reducor )        (const struct kaapi_format_t*, unsigned int, const void*, void*, const void*)
) : Format(0)
{
  if (fmt ==0)
    fmt = new kaapi_format_t;
  kaapi_format_taskregister_func( 
        fmt,
        0, 
        name.c_str(),
        size,
        get_count_params,
        get_mode_param,
        get_off_param,
        get_access_param,
        set_access_param,
        get_fmt_param,
/**/    get_size_param,
        reducor
  );
}



// --------------------------------------------------------------------------
template <> const WrapperFormat<char> WrapperFormat<char>::format(kaapi_char_format);
template <> const WrapperFormat<short> WrapperFormat<short>::format(kaapi_short_format);
template <> const WrapperFormat<int> WrapperFormat<int>::format(kaapi_int_format);
template <> const WrapperFormat<long> WrapperFormat<long>::format(kaapi_long_format);
template <> const WrapperFormat<unsigned char> WrapperFormat<unsigned char>::format(kaapi_uchar_format);
template <> const WrapperFormat<unsigned short> WrapperFormat<unsigned short>::format(kaapi_ushort_format);
template <> const WrapperFormat<unsigned int> WrapperFormat<unsigned int>::format(kaapi_uint_format);
template <> const WrapperFormat<unsigned long> WrapperFormat<unsigned long>::format(kaapi_ulong_format);
template <> const WrapperFormat<float> WrapperFormat<float>::format(kaapi_float_format);
template <> const WrapperFormat<double> WrapperFormat<double>::format(kaapi_double_format);
  
} // namespace ka
