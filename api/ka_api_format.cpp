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
  kaapi_format_register( this, strdup(fmt_name.c_str()));
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
#if 0
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
#endif

  template <> const Format* WrapperFormat<char>::get_format() { return (const Format*)&kaapi_char_format; }
  template <> const Format* WrapperFormat<short>::get_format() { return (const Format*)&kaapi_short_format; }
  template <> const Format* WrapperFormat<int>::get_format() { return (const Format*)&kaapi_int_format; }
  template <> const Format* WrapperFormat<long>::get_format() { return (const Format*)&kaapi_long_format; }
  template <> const Format* WrapperFormat<unsigned char>::get_format() { return (const Format*)&kaapi_uchar_format; }
  template <> const Format* WrapperFormat<unsigned short>::get_format() { return (const Format*)&kaapi_ushort_format; }
  template <> const Format* WrapperFormat<unsigned int>::get_format() { return (const Format*)&kaapi_uint_format; }
  template <> const Format* WrapperFormat<unsigned long>::get_format() { return (const Format*)&kaapi_ulong_format; }
  template <> const Format* WrapperFormat<float>::get_format() { return (const Format*)&kaapi_float_format; }
  template <> const Format* WrapperFormat<double>::get_format() { return (const Format*)&kaapi_double_format; }

#if 0
  template <> const Format* WrapperFormat<kaapi_int8_t>::get_format() { return (const Format*)&kaapi_char_format; }
  template <> const Format* WrapperFormat<kaapi_int16_t>::get_format() { return (const Format*)&kaapi_short_format; }
  template <> const Format* WrapperFormat<kaapi_int32_t>::get_format() { return (const Format*)&kaapi_int_format; }
  template <> const Format* WrapperFormat<kaapi_int64_t>::get_format() { return (const Format*)&kaapi_long_format; }
  template <> const Format* WrapperFormat<kaapi_uint8_t>::get_format() { return (const Format*)&kaapi_uchar_format; }
  template <> const Format* WrapperFormat<kaapi_uint16_t>::get_format() { return (const Format*)&kaapi_ushort_format; }
  template <> const Format* WrapperFormat<kaapi_uint32_t>::get_format() { return (const Format*)&kaapi_uint_format; }
  template <> const Format* WrapperFormat<kaapi_uint64_t>::get_format() { return (const Format*)&kaapi_ulong_format; }
#endif

const Format WrapperFormat<Access>::theformat(
  "Access",
  sizeof(Access),
  0,
  0,
  0,
  0,
  0,
  0
);
  
} // namespace ka
