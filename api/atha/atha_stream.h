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
#ifndef _ATHA_STREAM_H_
#define _ATHA_STREAM_H_

#include "atha_types.h"
#include "atha_error.h"
#include "atha_init.h"
#include <iosfwd>
#include <map>

namespace atha {
class Format;

// -------------------------------------------------------------------------
/** \name Stream_base
    \brief Base stream class for OStream and IStream
    \ingroup Serialization
*/
class Stream_base {
public:
  /** Virtual destructor
   */
  virtual ~Stream_base() {}

  /** Access mode when putting or getting data to/from the stream */
  enum Mode { 
    IA,   /* immediate access of value */ 
    DA,   /* differed access of value */
    DAC   /* differed access of possibly cyclic pointer value */
  };

  enum {
    THRESHOLD_IA = 32 /* threshold to pack putted data */
  }; 

  /** Type of Stream
      The type of a stream is given by a ORed values.
      Bit 0: 
        0 -> encode
        1 -> decode
      Bit 1:
        0 -> normal
        1 -> checkpoint
  */
  enum {
    ENCODE     = 0U,
    DECODE     = 1U,
    CHECKPOINT = 2U, 
    NORMAL     = 4U 
  };
  
  /** Return true iff the steam is of given type
      For instance is_type( DECODE | CHECKPOINT ) return true iff
      the stream is of kind 'checkpoint' with direction 'encode'.
  */
  bool is_type( uint8_t type ) const;

protected:
  /** Set the type of the stream
      Set the type of the stream using predefined Type value. OR of value
      is also possible, e.g. ENCODE | CHECKPOINT means that the type of
      the stream is for encoding into a checkpoint kind of stream.
      \param type a ORed value of type
  */
  void set_type( uint8_t type ) throw(InvalidArgumentError);

  /** Function called when the array of mode 'm' is fill.
    The return value is 'true' is a the stream process of defining
    a format could be continued, and 'false' is this process should
    be stop.
    @param m the mode of insertion into the stream that make overflow
    @param add_len the size requested at insertion
  */
  virtual bool overflow( Mode m, size_t add_len );

protected:
  struct {
    unsigned int _direction : 1;  /* 1 iff is an encode stream else is a decode stream */
    unsigned int _kind      : 1;  /* 1 iff is a checkpoint stream else is normal stream  */
  };
};


// -------------------------------------------------------------------------
/** \name OStream
    \brief OStream allows to define stream of data
    \ingroup Serialization
*/
class OStream : virtual public Stream_base {
public:
  /** Cstor 
  */
  OStream( );

  /** Dstor 
  */
  virtual ~OStream() {}
  
  /** Close a stream
  */
  virtual void close( );

  /** Write
      \param f [in] the format
      \param m [in] the mode of access (immediate or differed)
      \param data [in] the pointer to the data to communicate
      \param count [in] the number of formatted items to write
  */
  void write( const Format* f, Mode m, const void* data, size_t count );

  /** Write immediate data 
      \param f [in] the format
      \param data [in] the pointer to the data to communicate
      \param count [in] the number of formatted items to write
  */
  void write( const Format* f, const void* data, size_t count );

  /** Write unformatted bytes
      This method is called when all formatted data has been converted for
      heterogeneous communication.
      This method may be overloaded by derived class.
      \param m [in] the mode of access (immediate or differed)
      \param data [in] the pointer to the data to communicate
      \param count [in] the number of formatted items to write
  */
  virtual void write( Mode m, const void* data, size_t count ) =0;

  /** Context of the stream
  */
  typedef std::map<const void*,Pointer> Context;

  /** Find the definition of 'lptr'
      Find the identifier of 'ptr' and store it into 'pid'.
      \param lptr the pointer to find
      \param rptr the identifier of the pointer if exist
      \retval true iff it is the pointer is already bound or false
  */
  bool find( const void* lptr, Pointer& rptr);

  /** Bind a pointer to a new integer
      \param lptr the pointer to pack
      \param rptr the identifier of the pointer
      \retval true iff it is the first bind of this pointer
   */
  bool bind( const void* lptr, Pointer& rptr);
  
  /** Return the context object
  */
  Context& get_context();

protected:
  Context     _context;
  uint64_t    _current_id;
};


// -------------------------------------------------------------------------
/** \name IStream
    \brief IStream allows to define stream of data over a callstack
    \ingroup Serialization
*/
class IStream : virtual public Stream_base, public InterfaceAllocator {
public:
  /** Cstor 
  */
  IStream( Architecture a = Init::local_archi);

  /** Dstor 
  */
  virtual ~IStream() {}

  /** Close a stream
  */
  virtual void close( );
  
  /** Read
      \param f [in] the format
      \param m [in] the mode of access (immediate or differed)
      \param data [out] the pointer to the data to read
      \param count [in] the number of formatted items to read
  */
  void read( const Format* f, Mode m, void* const data, size_t count );

  /** Read immediate data
      \param f [in] the format
      \param data [out] the pointer to the data to read
      \param count [in] the number of formatted items to read
  */
  void read( const Format* f, void* const data, size_t count );

  /** Read unformatted bytes
      This method is called when all formatted data has been converted for
      heterogeneous communication.
      This method may be overloaded by derived class.
      \param m [in] the mode of access (immediate or differed)
      \param data [in] the pointer to the data to communicate
      \param count [in] the number of formatted items to write
  */
  virtual void read( Mode m, void* const data, size_t count ) = 0;

  /** Context of the stream
  */
  typedef std::map<Pointer,void*> Context;

  /** Find the pointer associated to 'pid'
      Find the pointer associated to 'pid' and store its value into 'ptr'.
      \param rptr the identifier to find
      \param lptr the value of pointer if found
      \retval true if the rptr is already bound or is equal to 0
  */
  bool find( Pointer rptr, void*& lptr);

  /** Bind an identifier to a pointer
      Bind to the identifier 'rptr' the pointer 'lptr'.
      \param rptr the identifier to bind
      \param lptr the value of pointer to bind
   */
  void bind( Pointer rptr, void* lptr);

  /** Return the context object
  */
  Context& get_context();
  
  /** To tie an allocator
  */
  //@{
  /** Tie the allocator used by the stream
      \param sa the allocator used by allocate method
      \retval the old Allocator
  */
  InterfaceAllocator* tie( InterfaceAllocator* sa );

  /** Allocate memory 
  */
  void* allocate( size_t sz );
  //@}

protected:
  /* contains set of pointer already inserted into the stream 
    _map_cylic[id] = ptr -> the key id has been already inserted 
    and ptr is the address of the memory that should contains the data.
  */
  Context             _context;
  Architecture        _archi;
  InterfaceAllocator* _sa;
};



// -------------------------------------------------------------------------
/** \name OStream2std_ostream  
    \brief OStream2std_ostream is a wrapper to build OStream over std::ostream
    \ingroup Serialization
*/
class OStream2std_ostream : public OStream {
public:
  /** Cstor */
  OStream2std_ostream( std::ostream* os )
   : OStream(), _s_out(os)
  { }

  /**
  */
  void open( std::ostream* os );

  /**
  */
  void close( );

  /** Write 
  */
  void write( Mode m, const void* data, size_t count );

protected:
  std::ostream* _s_out;
};


// -------------------------------------------------------------------------
/** \name IStream2std_istream  
    \brief IStream2std_istream is a wrapper to build IStream over std::istream
    \ingroup Serialization
*/
class IStream2std_istream : public IStream {
public:
  /** Cstor */
  IStream2std_istream( std::istream* is )
   : IStream(), _s_in(is)
  { }

  /**
  */
  void open( std::istream* is );

  /**
  */
  void close( );

  /** Read 
  */
  void read( Mode m, void* const data, size_t count );

protected:
  std::istream* _s_in;
};


// -------------------------------------------------------------------------
/** \name OStream2fd  
    \brief OStream2fd is a wrapper to build OStream over a file descriptor
    \ingroup Serialization
*/
class OStream2fd : public OStream {
public:
  /** Cstor */
  OStream2fd( int fd )
   : OStream(), _fd(fd)
  { }

  /**
  */
  void close( );

  /** Write 
  */
  void write( Mode m, const void* data, size_t count );

protected:
  int _fd;
};


// -------------------------------------------------------------------------
/** \name IStream2fd  
    \brief IStream2fd is a wrapper to build IStream over a file descriptor
    \ingroup Serialization
*/
class IStream2fd : public IStream {
public:
  /** Cstor */
  IStream2fd( int fd )
   : IStream(), _fd(fd)
  { }

  /**
  */
  void close( );

  /** Read 
  */
  void read( Mode m, void* const data, size_t count );

protected:
  int _fd;
};



/*
 * inline definition, see at the end of kaapi_format.h
 */
inline void OStream::close( )
{ }

inline void OStream::write ( const Format* f, const void* data, size_t count )
{ write( f, IA, data, count ); }

inline OStream::Context& OStream::get_context()
{ return _context; }

inline void IStream::close( )
{ }

inline void IStream::read ( const Format* f, void* const data, size_t count )
{ read( f, IA, data, count ); }

inline IStream::Context& IStream::get_context()
{ return _context; }

inline InterfaceAllocator* IStream::tie( InterfaceAllocator* sa )
{ 
  InterfaceAllocator* retval = _sa;
  _sa = sa; 
  return retval;
}

inline void OStream2std_ostream::open( std::ostream* os )
{ _s_out = os; }

inline void OStream2std_ostream::close( )
{ _s_out = 0; }
 
inline void OStream2fd::close( )
{ _fd = -1; }
 
inline void IStream2std_istream::open( std::istream* is )
{ _s_in = is; }

inline void IStream2std_istream::close( )
{ _s_in = 0; }
 
inline void IStream2fd::close( )
{ _fd = -1; }
 
} // namespace Util

#endif
