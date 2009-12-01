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
#include "atha_properties.h"
#include <iostream>
#include <sstream>
#include <ctype.h>

namespace atha {

// ---------------------------------------------------------------------------
//
// Skip any caracters until the end of the line
//
inline std::istream& skip_line(std::istream& s_in)
{
  char c =' ';
  while ( s_in.get(c) ) {
    if (c == '\n') return s_in;
    std::cout << "skip=" << c << std::endl;
  }
  s_in.putback(c);
  return s_in;
}


// ---------------------------------------------------------------------------
//
//
Properties::Properties() 
{}


// ---------------------------------------------------------------------------
Properties::~Properties()
{
  _map.clear();
}


// ---------------------------------------------------------------------------
void Properties::clear()
{ _map.clear(); }


// ---------------------------------------------------------------------------
void Properties::load(const std::string& fileName) throw (IOError)
{
  std::ifstream file;
  file.open( fileName.c_str() );
  if (!file.is_open()) Exception_throw(IOError("file not open") );
  if (!file.good()) Exception_throw(IOError("file not good") );

  std::string line;
  std::string key;
  std::string value;
  while( !file.eof() )
  {
    key = "";

    // - read a line ? read what ?
    file >> std::ws;
    std::getline(file, line);
    if (line == "") continue;
    
    // - skip comment
    if ((line[0] == '#') || (line[0] == '!') 
      || ((line[0] == '/') && (line[1] =='/'))) {
      continue; 
    }
    else {
      /* find the key part of the line */
      size_t ipos1 = line.find('=',0); // -get before '='
      if (ipos1 == std::string::npos) {
        std::cerr << "Bad format for line '" << line 
                  << "', should of format 'key=value;', cannot find '='" 
                  << std::endl;
        continue;
      }
      std::string tmp = line.substr(0,ipos1);
      //std::cout << "Key into='" << tmp << "'" << std::endl;
#if (KAAPI_GNUCXX==295)
      std::istrstream f_tmp(tmp.c_str());
#else
      std::istringstream f_tmp(tmp);
#endif
      f_tmp >> std::ws >> key;
    
      /* find the value part of the line */
      size_t ipos2 = line.find(';', ipos1 ); // -get before ';'
      if (ipos2 == std::string::npos) {
        std::cerr << "Bad format for line '" << line 
                  << "', should of format 'key=value;', cannot find ';'" 
                  << std::endl;
        continue;
      }
      tmp = line.substr( ipos1+1, ipos2-ipos1-1 );
      ipos1 = tmp.find_first_not_of(' ');
      value = tmp.substr( ipos1 );

      // assignement
      (*this)[key] = value;
    }
  }
  file.close();
}


// ---------------------------------------------------------------------------
//
//
void Properties::store(
  const std::string& filename, 
  const std::string& header) const throw (IOError)
{
  std::ofstream file;
  file.open( filename.c_str(), std::ios::out );
  if (!file.good()) Exception_throw( IOError() );

  file << "# " << header << std::endl;
  print( file );
  file.close();
}


// ---------------------------------------------------------------------------
//
//
void Properties::print( std::ostream& cout ) const
{
  std::map<std::string,std::string>::const_iterator curr = _map.begin(); 
  std::map<std::string,std::string>::const_iterator end = _map.end(); 
  while (curr != end) {
    if ( (*curr).second != "") 
      cout << (*curr).first << "=" << (*curr).second << ';' << std::endl;
    ++curr;
  }
}


// ---------------------------------------------------------------------------
//
//
std::string&  Properties::operator[]( const std::string& key)
{
  return _map[key];
}

const std::string&  Properties::operator[]( const std::string& key) const
{
  // not a operator[](...) const  on map ???
  return ((std::map<std::string,std::string>&)_map)[key];
}

void Properties::insert( const std::string& key, const std::string& value)
{
  std::pair<iterator,bool> retval;
  retval = _map.insert( std::make_pair(key,value) );
  if (retval.second ==true) return;
  retval.first->second = value;
}


// ---------------------------------------------------------------------------
Properties::iterator Properties::find( const std::string& key )
{
  return _map.find(key);
}

// ---------------------------------------------------------------------------
//
void Properties::merge( const Properties& prop )
{
  const_iterator ibegin = prop.begin();
  const_iterator iend   = prop.end();
  while (ibegin != iend)
  {
    (*this)[ibegin->first] = ibegin->second;
    ++ibegin;
  }
}

} //namespace
