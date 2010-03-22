/*
** xkaapi
** 
** Created on Tue Mar 31 15:19:14 2009
** Copyright 2009 INRIA.
**
** Contributors :
**
** thierry.gautier@inrialpes.fr
** xavier.besseron@imag.fr
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
#include "atha_init.h"
#include "atha_format.h"
#include <iostream>


// standard size used to transfer data on the network
#define STD_SIZE_BOOL       1

#define STD_SIZE_CHAR       1
#define STD_SIZE_SHORT      2
#define STD_SIZE_INT        4
#define STD_SIZE_LONG       8
#define STD_SIZE_LONGLONG   8

#define STD_SIZE_FLOAT      4
#define STD_SIZE_DOUBLE     8
#define STD_SIZE_LONGDOUBLE 16

/*
*/
namespace atha {

/* -----------------------------------
*/
std::map<Format::Id,Format*> Format::_all_fm; // map from Id to format object
Format* Format::_base_fmt =0;                 // base format to initialize

// --------------------------------------------------------------------
Format::~Format()
{
}


// --------------------------------------------------------------------
Format::Format(
    Id fmid,
    size_t sz,
    const std::string& n
)
  : _id( fmid ),
    _name(n),
    _size(sz),
    _iscompiled(false),
    _iscontiguous(false)
{
//  Format* fmt_prev = get_format( _id );
//  KAAPI_ASSERT_M (fmt_prev ==0, "[Format::Format] already defined entry: collision of two object with the same name");
  // link to postponed storage in the map
  _prev_init = _base_fmt;
  _base_fmt = this;
  //_all_fm.insert( std::make_pair(_id,this));
}


// --------------------------------------------------------------------
Format::Format( const Format& )
{}


// --------------------------------------------------------------------
Format& Format::operator=( const Format& )
{
  return *this;
}


// --------------------------------------------------------------------
FormatOperationCumul::FormatOperationCumul(
    Format::Id   fid,
    size_t             sz,
    const std::string& name
  )
 : Format(fid, sz, name)
{
}


// --------------------------------------------------------------------
void ByteFormat::write( OStream& s, const void* val, size_t count ) const
{
  s.write( &FormatDef::Byte, OStream::DA, val, count);
}


// --------------------------------------------------------------------
void ByteFormat::read( IStream& s, void* val, size_t count ) const
{
  s.read( &FormatDef::Byte, IStream::DA, val, count);
}


// --------------------------------------------------------------------
void Format::print( std::ostream& o, const void*  ) const
{ o << "<>"; }


// --------------------------------------------------------------------
namespace FormatDef {
  const NullFormat Null;
  const ByteFormat Byte;
}


// --------------------------------------------------------------------
KAAPI_DECL_SPEC_FORMAT(bool, FormatDef::Bool)
KAAPI_DECL_SPEC_FORMAT(char, FormatDef::Char)
KAAPI_DECL_SPEC_FORMAT(signed char, FormatDef::SChar)
KAAPI_DECL_SPEC_FORMAT(unsigned char, FormatDef::UChar)
KAAPI_DECL_SPEC_FORMAT(int, FormatDef::Int)
KAAPI_DECL_SPEC_FORMAT(unsigned int, FormatDef::UInt)
KAAPI_DECL_SPEC_FORMAT(short, FormatDef::Short)
KAAPI_DECL_SPEC_FORMAT(unsigned short, FormatDef::UShort)
KAAPI_DECL_SPEC_FORMAT(long, FormatDef::Long)
KAAPI_DECL_SPEC_FORMAT(unsigned long, FormatDef::ULong)
KAAPI_DECL_SPEC_FORMAT(long long, FormatDef::LLong)
KAAPI_DECL_SPEC_FORMAT(unsigned long long, FormatDef::ULLong)
KAAPI_DECL_SPEC_FORMAT(float, FormatDef::Float)
KAAPI_DECL_SPEC_FORMAT(double, FormatDef::Double)
KAAPI_DECL_SPEC_FORMAT(long double, FormatDef::LDouble)


// --------------------------------------------------------------------
void Format::init_format()
{
  static bool once=true; 
  if (once) once=false; 
  else return;

  FormatDef::Byte.set_continuous();
  FormatDef::Bool.set_continuous();
  FormatDef::Char.set_continuous();
  FormatDef::SChar.set_continuous();
  FormatDef::UChar.set_continuous();
  FormatDef::Int.set_continuous();
  FormatDef::UInt.set_continuous();
  FormatDef::Short.set_continuous();
  FormatDef::UShort.set_continuous();
  FormatDef::Long.set_continuous();
  FormatDef::ULong.set_continuous();
  FormatDef::LLong.set_continuous();
  FormatDef::ULLong.set_continuous();
  FormatDef::Float.set_continuous();
  FormatDef::Double.set_continuous();
  FormatDef::LDouble.set_continuous();
  FormatDef::Null._id =0;

  while (_base_fmt !=0) 
  {
    KAAPI_CPPLOG(Init::verboseon, "[Format::init_format] Register format: id=" << _base_fmt->get_id() 
       << ", name=" << _base_fmt->get_name());
    Format* fmt = Format::get_format(_base_fmt->get_id());
    if (fmt !=0) {
      std::ostringstream msg;
      msg << "[Format::init_format] collision detected for hash value of name '" 
          << fmt->get_name() << "' and name '" << _base_fmt->get_name() 
          << "', you should change the hash function hash_value_str in utils_types.cpp";
      System_abort( msg.str() );
    }
    _all_fm.insert( std::make_pair(_base_fmt->get_id(),_base_fmt));
    _base_fmt = _base_fmt->_prev_init;
  }
}


// --------------------------------------------------------------------
void Format::set_name( const std::string& n )
{
  _name = n;
}


// --------------------------------------------------------------------
const std::string& Format::get_name( ) const
{
  return _name;
}


} // namespace
