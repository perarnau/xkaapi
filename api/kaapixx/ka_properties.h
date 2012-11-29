/*
** xkaapi
** 
**
** Copyright 2009,2010,2011,2012 INRIA.
**
** Contributors :
**
** thierry.gautier@inrialpes.fr
** xavier.besseron@imag.fr
** vincent.danjean@imag.fr
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
#ifndef _KA_PROPERTIES_H_
#define _KA_PROPERTIES_H_

#include <fstream>
#include <string>
#include <map>
#include <stdexcept>
#include "ka_error.h"

namespace ka {

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
  virtual void load( const std::string& filename ) throw(std::runtime_error);

  /** Store the file
      \exception IOError if cannot store the file
  */
  virtual void store( const std::string& filename, 
              const std::string& header = "" ) const throw(std::runtime_error);

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
