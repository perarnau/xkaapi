// ==========================================================================
// (c) INRIA, projet MOAIS, 2006-2009
// Author: Thierry Gautier
// Status: ok
// ==========================================================================

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
