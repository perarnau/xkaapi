// =========================================================================
// (c) INRIA, projet MOAIS, 2006
// Author: T. Gautier, X. Besseron
//
//
// =========================================================================
#include "kaapi_impl.h"
#include "atha_format.h"
#include "atha_stream.h"
#include "atha_init.h"

#include <stdlib.h>
#include "atha_error.h"

#include <unistd.h>
#include <iostream>

namespace atha {

#define MASK_ALLTYPE    0x7
#define MASK_DIRECTION  0x1
#define MASK_CHECKPOINT 0x2
#define MASK_NORMAL     0x4

// --------------------------------------------------------------------
void Stream_base::set_type( kaapi_uint8_t type ) throw(InvalidArgumentError)
{
  if ((type & MASK_ALLTYPE) !=0)
    Exception_throw(InvalidArgumentError("bad type"));
  if (((type & MASK_CHECKPOINT) !=0) && ((type & MASK_NORMAL) !=0))
    Exception_throw(InvalidArgumentError("bad type"));

  if ( (type & MASK_DIRECTION) !=0)
    _direction = 1U;
  else
    _direction = 0U;

  if ( (type & ~MASK_CHECKPOINT) !=0)
    _kind = 1U;
  else
    _kind = 0U;
}


// --------------------------------------------------------------------
bool Stream_base::is_type( kaapi_uint8_t type ) const
{
  bool retval = false;
  retval |= (((type & MASK_DIRECTION) ? 0U : 1U) == _direction);
  retval &= (((type & MASK_DIRECTION) ? 0U : 1U) == _direction);
  return retval;
}


// --------------------------------------------------------------------
bool Stream_base::overflow( Mode , size_t )
{
  std::cerr << "[Stream_base::overflow] cannot put or pack more data" << std::endl;
  return false;
}


// --------------------------------------------------------------------
OStream::OStream( ) 
  : Stream_base(), _context(), _current_id(1)
{ 
  _context.clear();
}


// --------------------------------------------------------------------
#define STD_LONGDOUBLE_FORMAT IEEE_QUADRUPLE

#define DEBUG_XDR_WRITE false
#define DEBUG_XDR_READ  false
#define DEBUG_ASN false

// --------------------------------------------------------------------
void affiche_mem(const char *s, const unsigned char* ad, unsigned int n){
  printf("%s - ",s);
  for (unsigned int i = 0 ; i < n ; i++){
    printf("%02x ", (unsigned char)ad[i]);
  }
  printf("\n");
}


// --------------------------------------------------------------------
void OStream::write ( const Format* f, Mode m, const void* data, size_t count )
{
  size_t size = f->get_size();
  size_t size_byte = count*size;
#ifndef KAAPI_USE_XDR
  KAAPI_CPPLOG( DEBUG_ASN, "[atha::OStream::write] ASN::write : count*size =" << count << "*" << size);
  if ((m == OStream::IA) || (size_byte <16))
    write( OStream::IA, data, size_byte );
//    _msg->pack( data, size_byte );
  else 
    write( OStream::DA, data, size_byte );
//    _msg->put( data, size_byte );

#else // #ifndef KAAPI_USE_XDR
  size_t std_size = f->get_std_size();
  if (std_size == 0) 
  {   // Raw data
    if ((m == OStream::IA) || (size_byte <16))
      write( OStream::IA, data, size_byte );
      //_msg->pack( data, size_byte );
    else
      write( OStream::IA, data, size_byte );
      //_msg->put( data, size_byte );

  } else {    // Typed data
    /**** Alloc and copy ****/
    unsigned char *d = NULL;
    if (std_size == size) 
    {
#ifdef WORDS_BIGENDIAN
      d = (unsigned char *) calloc(count, std_size);
      memcpy(d, data, std_size*count);
#else
      if (LongDouble::get_local_format() == STD_LONGDOUBLE_FORMAT) 
      {
        d = (unsigned char *) data;
      } else {
        d = (unsigned char *) calloc(count, std_size);
        memcpy(d, data, std_size*count);
      }
#endif
    } 
    else if (size < std_size) 
    {
      d = (unsigned char *) calloc(count, std_size);
      for (unsigned int i = 0 ; i < count ; i++)
        memcpy( d + (i * std_size) , (unsigned char*) data + (i * size) , size);
    } else {
      d = (unsigned char *) calloc(count, std_size);
      for (unsigned int i = 0 ; i < count ; i++)
#ifdef WORDS_BIGENDIAN
        memcpy( d + (i * std_size) , (unsigned char*) data + (i * size) + (size - std_size), std_size);
#else
        memcpy( d + (i * std_size) , (unsigned char*) data + (i * size) , std_size);
#endif
    }
    KAAPI_ASSERT_M( d != NULL , "Pointer not initialized");

    /**** Swap ****/
#ifdef WORDS_BIGENDIAN
      // Big Endian
      if (size <= std_size){
        for (unsigned int i = 0 ; i < count ; i++)
          ByteSwap( d + (i * std_size) , size);
      } else {
        for (unsigned int i = 0 ; i < count ; i++)
          ByteSwap( d + (i * std_size) , std_size);
      }
#endif

    /**** Conversion ****/
    if (f->is_real())
    {
      if (f->get_id() == FormatDef::LDouble.get_id())
      {
        for (unsigned int i = 0 ; i < count ; i++) {
          if (DEBUG_XDR_WRITE) affiche_mem("\nwrite - Avant conversion",d + (i * std_size), std_size);
          LongDouble::conversion_from_local_to(STD_LONGDOUBLE_FORMAT, d + (i * std_size));
          if (DEBUG_XDR_WRITE) affiche_mem("write - Apres conversion",d + (i * std_size), std_size);
        }
      }
    } 
    else if (!(f->is_unsigned())) 
    {
      if (size < std_size) 
      {
        // complete the number
        for (unsigned int i = 0 ; i < count ; i++) 
        {
          if (((d + (i * std_size))[size-1] & 0x80) == 0x80) {
          // negative signed integers
            for (unsigned int j = size ; j < std_size ; j++){
              *(d + (i * std_size) + j) = 0xff;
            }
          }
        }
      }
    }

    /**** pack ****/
    write( OStream::IA, d, std_size * count );
    //_msg->pack( d, std_size * count );

    /**** Free ****/
#ifdef WORDS_BIGENDIAN
    // Big Endian
    delete d;
#else
    // Little Endian
    if ((std_size != size) || (LongDouble::get_local_format() != STD_LONGDOUBLE_FORMAT))
    {
      delete d;
    }
#endif
  }
#endif
}


// --------------------------------------------------------------------
IStream::IStream( Architecture a )
  : Stream_base(), _context(), _archi(a)
{
  _context.clear();
}


// --------------------------------------------------------------------
void IStream::read ( const Format* f, Mode m, void* const data, size_t count )
{
#ifdef KAAPI_USE_ASN
  size_t local_size = f->get_size();
  size_t sent_size;
  
  size_t sent_sz_b  =   _archi.get_sizeof_bool();
  size_t sent_sz_l  =   _archi.get_sizeof_long();
  LongDoubleFormat sent_fmt_ld = _archi.get_formatof_longdouble();
  size_t sent_sz_ld =   LongDouble::get_size(sent_fmt_ld);
  bool sent_bigendian = _archi.is_bigendian();

  bool is_bool = f->get_id() == FormatDef::Bool.get_id();
  bool is_long = (f->get_id() == FormatDef::Long.get_id()) || (f->get_id() == FormatDef::ULong.get_id());
  bool is_ldouble = f->get_id() == FormatDef::LDouble.get_id();

  

  if (   ((sent_sz_b  == sizeof(bool)) || (!is_bool))
      && ((sent_sz_l  == sizeof(long)) || (!is_long))
      && ((sent_sz_ld == LongDouble::get_size(LongDouble::get_local_format())) || (!is_ldouble))  ) 
  {
    KAAPI_CPPLOG( DEBUG_ASN, "[atha::IStream::read] ASN::read : sent_size = local_size");

    /**** Get or Unpack ****/
    if ((m == IStream::IA) || (local_size*count <16)) 
    {
      read( IStream::IA, data, local_size*count );
      //_cc->unpack( data, local_size*count );
    } else {
      read( IStream::DA, data, local_size*count );
      //_cc->get( data, local_size*count );
    }
    KAAPI_CPPLOG( DEBUG_ASN, "[atha::IStream::read] ASN::read : count*size = " << count << "*" << local_size);

    // long double conversion must be done in little-endian
    if (is_ldouble && (sent_fmt_ld != LongDouble::get_local_format())) 
    {
      // Swap to little-endian
      if (sent_bigendian) {                       
        for (unsigned int i = 0 ; i < count ; i++)
          ByteSwap( (unsigned char *)data + (i * local_size) , local_size);
      }
      // Conversion
      for (unsigned int i = 0 ; i < count ; i++){
        LongDouble::conversion_to_local_from(sent_fmt_ld, (unsigned char*)data + (i*local_size));
      }
      // Swap to big-endian if needed
#ifdef WORDS_BIGENDIAN
        for (unsigned int i = 0 ; i < count ; i++)
          ByteSwap( (unsigned char *)data + (i * local_size) , local_size);
#endif          
    } else { 
      // Swap only if needed      
      /**** Swap ****/
#ifdef WORDS_BIGENDIAN
        if (!sent_bigendian) {                       
          for (unsigned int i = 0 ; i < count ; i++)
            ByteSwap( (unsigned char *)data + (i * local_size) , local_size);
        }
#else
        if (sent_bigendian) {
          for (unsigned int i = 0 ; i < count ; i++)
            ByteSwap( (unsigned char *)data + (i * local_size) , local_size);
        }
#endif
    }

  } else {
    if (is_bool) {
      sent_size = sent_sz_b;
    } else if (is_long) {
      sent_size = sent_sz_l;
    } else if (is_ldouble) {
      sent_size = sent_sz_ld;
    } else {
      sent_size = local_size;
    }
    KAAPI_ASSERT_M( local_size != sent_size , "Bad assertion");

    if (sent_size > local_size)
    {
      KAAPI_CPPLOG( DEBUG_ASN, "[atha::IStream::read] ASN::read : sent_size > local_size");
      /**** sent_size > local_size --- Alloc ****/
      unsigned char *d =  new unsigned char[count * sent_size];
    
      /**** sent_size > local_size --- Get or Unpack ****/
      if ((m == IStream::IA) || (sent_size*count <16))
      {
        read( IStream::IA, d, sent_size*count );
        //_cc->unpack( d, sent_size*count );
      } else {
        read( IStream::DA, d, sent_size*count );
        //_cc->get( d, sent_size*count );
      }
      KAAPI_CPPLOG( DEBUG_ASN, "[atha::IStream::read] ASN::read : count*size = " << count << "*" << sent_size);

      /**** sent_size > local_size --- Swap ****/
      if (sent_bigendian) {
        for (unsigned int i = 0 ; i < count ; i++)
          ByteSwap( d + (i * sent_size) , sent_size);
      }

      /**** sent_size > local_size --- Conversion ****/
      if (is_ldouble) {
        for (unsigned int i = 0 ; i < count ; i++){
          LongDouble::conversion_to_local_from(sent_fmt_ld, d + (i * sent_size));
        }
      } else if (is_long) {
        KAAPI_ASSERT_M( sent_size == 8 && local_size == 4, "Bad assertion");
        // Nothing to do
      } else if (is_bool) {
        KAAPI_ASSERT_M( sent_size == 4 && local_size == 1, "Bad assertion");
        // Nothing to do
      } 

      /**** sent_size > local_size --- Swap ****/
#ifdef WORDS_BIGENDIAN
        for (unsigned int i = 0 ; i < count ; i++)
          ByteSwap( d + (i * sent_size) , local_size);
#endif
      
      /**** sent_size > local_size --- Copy then free ****/
      for (unsigned int i = 0 ; i < count ; i++){
          memcpy( (unsigned char*)data + (i * local_size), d + (i * sent_size), local_size);
      }
      delete d;

    } else 
    {
      KAAPI_CPPLOG( DEBUG_ASN, "[atha::IStream::read] ASN::read : sent_size < local_size");

      /**** sent_size < local_size --- Get or Unpack ****/
      if ((m == IStream::IA) || (sent_size*count <16))
      {
        for (unsigned int i = 0 ; i < count ; i++)
          read( IStream::IA, (unsigned char*)data + (i * local_size), sent_size );
          //_cc->unpack( (unsigned char*)data + (i * local_size), sent_size );
      } else {
        for (unsigned int i = 0 ; i < count ; i++)
          read( IStream::DA, (unsigned char*)data + (i * local_size), sent_size );
          //_cc->get( (unsigned char*)data + (i * local_size), sent_size );
      }
      KAAPI_CPPLOG( DEBUG_ASN, "[atha::IStream::read] ASN::read : count*size = " << count << "*" << sent_size);

      /**** sent_size < local_size --- Swap ****/
      if (sent_bigendian) 
      {
        for (unsigned int i = 0 ; i < count ; i++)
          ByteSwap( (unsigned char*) data + (i * local_size) , sent_size);
      }

      /**** sent_size < local_size --- Conversion ****/
      if (is_ldouble) 
      {
        for (unsigned int i = 0 ; i < count ; i++){
          LongDouble::conversion_to_local_from(sent_fmt_ld, (unsigned char*)data + (i*local_size));
        }

      } 
      else if (is_long) 
      {
        KAAPI_ASSERT_M( local_size == 8 && sent_size == 4, "Bad assertion");
        if (f->get_id() == FormatDef::Long.get_id()) 
        {
          // signed long
          for (unsigned int i = 0 ; i < count ; i++) {
            if ((((unsigned char*)data + (i * local_size))[sent_size-1] & 0x80) == 0x80) {
              // negative signed long
              memset((unsigned char*)data + (i * local_size) + sent_size, 0xff, local_size - sent_size);
            } else {
              // positive signed long
              memset((unsigned char*)data + (i * local_size) + sent_size, 0x00, local_size - sent_size);
            }
          }
        } else {
          // unsigned long
          for (unsigned int i = 0 ; i < count ; i++) {
            memset((unsigned char*)data + (i * local_size) + sent_size, 0x00, local_size - sent_size);
          }
        }
      } else if (is_bool) 
      {
        KAAPI_ASSERT_M( local_size == 4 && sent_size == 1, "Bad assertion");
        for (unsigned int i = 0 ; i < count ; i++) {
          memset((unsigned char*)data + (i * local_size) + sent_size, 0x00, local_size - sent_size);
        }
      }

      /**** sent_size < local_size --- Swap ****/
#ifdef WORDS_BIGENDIAN
        for (unsigned int i = 0 ; i < count ; i++)
          ByteSwap( (unsigned char*)data + (i * local_size) , local_size);
#endif

    }
  }

#elif KAAPI_USE_XDR

  size_t size = f->get_size();
  size_t std_size = f->get_std_size();
  if (std_size == 0) 
  {   // Raw data
    if ((m == IStream::IA) || (size <16))
      read( IStream::IA, data, size*count );
      //_cc->unpack( data, size*count );
    else
      read( IStream::DA, data, size*count );
      //_cc->get( data, size*count );
  } 
  else 
  {    // Typed data
    /**** Alloc ****/
    unsigned char *d = NULL;
    if (std_size == size)
    {
      d = (unsigned char *) data;
    } else {
      d = new unsigned char[count * std_size];
    }
    KAAPI_ASSERT_M( d != NULL , "Pointer not initialized");

    /**** unpack ****/
    read( IStream::IA, d, std_size * count );
    //_cc->unpack( d, std_size * count );

    /**** Conversion ****/
    if (f->is_real())
    {
      if (f->get_id() == FormatDef::LDouble.get_id())
      {
        for (unsigned int i = 0 ; i < count ; i++) {
          if (DEBUG_XDR_READ) affiche_mem("\nread - Avant conversion",d + (i * std_size), std_size);
          LongDouble::conversion_to_local_from(STD_LONGDOUBLE_FORMAT, d + (i * std_size));
          if (DEBUG_XDR_READ) affiche_mem("read - Apres conversion",d + (i * std_size), std_size);
        }
      }
    }

    /**** Swap ****/
#ifdef WORDS_BIGENDIAN
    // Big Endian
    if (size <= std_size) {
      for (unsigned int i = 0 ; i < count ; i++)
        ByteSwap( d + (i * std_size) , size);
    } else {
      for (unsigned int i = 0 ; i < count ; i++)
        ByteSwap( d + (i * std_size) , std_size);
    }
#endif

    /**** Copy then Free ****/
    if (size < std_size){
      for (unsigned int i = 0 ; i < count ; i++)
        memcpy( (unsigned char*)data + (i * size) , d + (i * std_size) , size);
      delete d;
    } else if (size > std_size) {
      for (unsigned int i = 0 ; i < count ; i++){
#ifdef WORDS_BIGENDIAN
          memset((unsigned char*)data + (i * size), 0, (size - std_size));
          memcpy( (unsigned char*)data + (i * size) + (size - std_size), d + (i * std_size) , std_size);
#else
          memcpy( (unsigned char*)data + (i * size) , d + (i * std_size) , std_size);
          memset((unsigned char*)data + (i * size) + std_size, 0, (size - std_size));
#endif
      }
      delete d;
    }
  }

#else
  size_t sz = count*f->get_size();
  if ((m == IStream::IA) || (sz <16))
    read( IStream::IA, data, sz );
    //_cc->unpack( data, sz );
  else
    read( IStream::DA, data, sz );
    //_cc->get( data, sz );
#endif
}



// --------------------------------------------------------------------
void* IStream::allocate( size_t sz )
{ return (_sa ==0 ? 0 : _sa->allocate(sz)); }


// --------------------------------------------------------------------
void OStream2std_ostream::write( Mode , const void* data, size_t count )
{
  _s_out->write( (const char*)data, count );
}


// --------------------------------------------------------------------
void IStream2std_istream::read( Mode , void* const data, size_t count )
{
  _s_in->read( (char*)data, count );
}


// --------------------------------------------------------------------
void OStream2fd::write( Mode , const void* data, size_t count )
{
  ::write( _fd, (const char*)data, count );
}


// --------------------------------------------------------------------
void IStream2fd::read( Mode , void* const data, size_t count )
{
  int sz = ::read( _fd, (char*)data, count );
  if (sz ==0) throw EndOfFile();
  if (sz == -1) throw IOError("Error", errno );
}


// --------------------------------------------------------------------
bool OStream::find( const void* ptr, Pointer& pid)
{
  if (ptr ==0) {
    pid = 0;
    return true;
  }
  Context::iterator curr = _context.find( ptr );
  if (curr == _context.end()) return false;
  pid = curr->second;
  return true;
}


// --------------------------------------------------------------------
bool OStream::bind( const void* ptr, Pointer& pid)
{
  if (ptr ==0) {
    pid = 0;
    return false;
  }
  Context::iterator curr = _context.find( ptr );
  if (curr == _context.end()) {
    pid.ptr = _current_id;
    ++_current_id;
    _context.insert( std::make_pair( ptr, pid) );
    return true;
  }
  pid = curr->second;
  return false;
}


// --------------------------------------------------------------------
bool IStream::find( Pointer pid, void*& ptr)
{
  if (pid ==0ULL) {
    ptr = 0;
    return true;
  }
  Context::iterator curr = _context.find( pid );
  if (curr == _context.end()) return false;
  ptr = curr->second;
  return true;
}


// --------------------------------------------------------------------
void IStream::bind( Pointer pid, void* ptr)
{
  if (pid != (void*)0)
    _context.insert( std::make_pair( pid, ptr) );
}

} // namespace Net

