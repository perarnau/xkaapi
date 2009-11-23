/* KAAPI public interface */
// ==========================================================================
// (c) INRIA, projet MOAIS, 2006-2009
// Author: T. Gautier
//
// ==========================================================================
#ifndef _ATHA_PROPERTIES_H_
#define _ATHA_PROPERTIES_H_

#include <fstream>
#include <string>
#include <map>
#include "atha_error.h"

namespace atha {

// ---------------------------------------------------------------------------
/** \class Properties
    \brief Properties class
    \ingroup atha
    Properties class allows to store as string parameters such as 'key' = 'value'.
    The class allows saving and loading Properties to/from files.
*/
// ---------------------------------------------------------------------------
class Properties {
public:

  /** Default Constructor
  */
  Properties();

  /** Destructor
  */
  virtual ~Properties();

  /** Clear all entries in the object
  */
  void clear();
  
  /** Load the file with name 'file'
      \exception IOError if cannot load the file
  */
  virtual void load( const std::string& filename ) throw(IOError);

  /** Store the file
      \exception IOError if cannot store the file
  */
  virtual void store( const std::string& filename, 
              const std::string& header = "" ) const throw(IOError);

  /** Print the set of parameters for the simulation
  */
  virtual void print( std::ostream& ) const;

  /** Return the value with 'key' or a null string if doesn't exist
  */
  std::string& operator[]( const std::string& key);

  /** Return the value with 'key' or a null string if doesn't exist
  */
  const std::string& operator[]( const std::string& key) const;

  /** Return the size of the properties = the number of properties
  */
  size_t size() { return _map.size(); }

  /** Iterator over the contents of the properties
  */
  //@{
  typedef std::map<std::string,std::string>::iterator iterator;
  typedef std::map<std::string,std::string>::const_iterator const_iterator;
  ///
  iterator begin() { return _map.begin(); }
  ///
  iterator end() { return _map.end(); }
  ///
  const_iterator begin() const { return _map.begin(); }
  ///
  const_iterator end() const { return _map.end(); }
  //@}
  
  /** Insert the key=value
      If key already exists, then current value is replaced by new value
  */
  void insert( const std::string& key, const std::string& value);

  /** Find the key
  */
  iterator find( const std::string& key );

  /** Merge the properties object src into this
  */
  void merge( const Properties& prop );

protected:  
  std::map<std::string,std::string> _map;
};

} // namespace

#endif // PROPERTY 
