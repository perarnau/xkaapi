/*
** xkaapi
** 
**
** Copyright 2009,2010,2011,2012 INRIA.
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
#ifndef _ATHA_PARSER_H_
#define _ATHA_PARSER_H_

#include "ka_properties.h"
#include <stdint.h>
#include <stdexcept>
#include <map>
#include <list>

namespace ka {

/** Paser class
    Paser class is used to parse command line arguments in order to
    form properties for a fixed set of predefined module name.
    The grammar of the command line is the following.
    moduleoptions ->  '--' modulename options | moduleoptions | {}
    options -> '-'optionname  | '-'optionname '=' values | options | {}
    modulename -> string
    optionname -> string
    values -> string
    string cannot contained '--' or declared options with '-'.

    Whatever is the rule for an option, the default value, if not specify
    on the command line, is 'true'. See example.
    
    Recognized command line aruments are deleted from the command line.
    The names for modules and options should be declared before parsing
    the command line and should not appear in the string for values.
    The output of the parser is for each declared module a properties
    object that contains strings :
    * for options with values :
      modulename.optionname=values
    * for options without values :
      modulename.optionname=true if the option is defined
    or
      modulename.optionname=false if the option is not defined

    Important remarks. The character '_' in optioname is transformed into '.' 
    in the properties object. For instance 'dummy_option1' defined in modulename 
    has a key 'modulename.dummy.options=...'.

    \verbatim
    // My parser will recognized :
    //   --shuttle -velocity 123 -gravity 9.3 -wind -12 
    // as well as 
    //   --shuttle -velocity 123 --toto -gravity 9.3 -wind -12
    // The output display should be (may be in a different order):
    //      shuttle.velocity=123;
    //      shuttle.gravity=9.3;
    //      shuttle.gravity=-12;
    // 
    // While the command line:
    // --shuttle -order 3
    // Produce the following output using '500' as default value for velocity.
    //      shuttle.velocity=500;
    //      shuttle.order=3;
    //
    // Note that command line 
    // --shuttle -order 
    // Will produce the following output:
    //      shuttle.order=true;
    // Because no value has been specify for the order !

    Parser my_parser;
    Properties my_prop;
    
    // form the modules
    Parser::Module shuttle_module("shuttle");
    // option with default value "500"
    shuttle_module.add_option("shuttle", "velocity","500");
    // option without default value
    shuttle_module.add_option("shuttle", "gravity");
    // option without default value
    shuttle_module.add_option("shuttle", "wind");
    // option with default value "2" and the usage for this option.
    shuttle_module.add_option("shuttle", "order", "2", "define the order of the numerical scheme for integration");

    // Add rules to the parser
    my_parser.add_module(shuttle_module, my_prop);

    // parse the command line    
    my_parser.parse( prop, argc, argv );
    
    // display output
    my_prop.print( std::cout );
    \endverbatim
*/
class Parser {
public:
  /** 
  */
  Parser();

  /** 
  */
  ~Parser();
  
  /** Definition of the rules that form a module 
  */
  class Module {
  public:
    /** Default constructor
    */
    Module();

    /** Constructor with a name for a module
    */
    Module(const std::string& name );

    /** Return the name of the module 
    */
    const std::string& get_name() const;

    /** Set the name of a module
    */
    void set_name(const std::string& );

    /** Set the helper function for this module
    */
    void set_helper( void (*helper)(std::ostream& o) );

    /** Declare a new option name for the module 
        \param name the option name 
        \param value the default value [optional]
        \param usage the usage of this option [optional]
        \param mode 'w' to use the last defined option, 'a' to append definition
        \param separator is the string to separate two options
    */
    void add_option( const std::string& name, 
                     const std::string& value = "<no value>", 
                     const std::string& usage = "[no usage]",
                     char mode = 'w',
                     const std::string& separator = ""
                   );
    
    /** Display usage informations about options of the module
    */
    void print_usage( std::ostream& o ) const;

    /** Dump default value for rc 
    */
    void dump_rc( std::ostream& o ) const;

  protected:
    friend class Parser;
    std::string get_resource_name(const std::string& option_name) const;
    /**/
    struct OptionInfo {
      OptionInfo( const std::string& n, const std::string& dv ="<no value>", const std::string& us = "[no usage]", 
                  char m = 'w', const std::string& s = "")
       : name(n), default_value(dv), usage(us), mode(m), separator(s)
      {}

      std::string name;
      std::string default_value;
      std::string usage;
      char        mode;
      std::string separator;
    };
    
    /**/
    class Options : public std::map<std::string,OptionInfo> { /* name -> val init */
    public:
      static std::string get_canonical_name(const std::string& option_name);
    };

    Options     options;
    std::string name;
    void      (*helper)(std::ostream& o);
  };

  /** Parse the argc, argv
      A new command line without recognized arguments are accessible from
      the new command line generated after parsing through the use of
      new_argc() and new_argv() methods.
      All recognized options and their values are stored into the properties
      object associated with the module name when defined.
  */
  void parse( Properties& initialprop, int& argc, char**& argv );
  
  /** Add the rules of parsing from an existing parser
      \param p the parser
  */
  void add_parser(const Parser& p);

  /** Add rules from a module.
      After call to this method, additional options may be added on 
      the returned object but not of the pass by value 'mod' object.
      \param mod the module
      \param prop the properties object where to store recognized options for this module
      \exception throw InvalidArgumentError if module has no name
  */
  void add_module( const Module* mod, Properties* prop)
    throw(std::invalid_argument);

  /** Declare a new module with a name 
      \param name the module name 
      \param prop the properties object where to store recognized options for this module
      \exception throw InvalidArgumentError if module is already added
  */
  void add_module(const std::string& name, Properties* prop)
    throw(std::invalid_argument);

  /** Display usage informations about all added modules
  */
  void print_usage( std::ostream& o ) const;

  /** Dump predefined values
  */
  void dump_rc( std::ostream& o ) const;

  /** Conversion methods from some basic type to string and vis et versa
  */
  //@{
  /** Convert a bool to a string
  */
  static const std::string& Bool2String( bool v );

  /** Convert a string to a bool
  */
  static bool String2Bool( const std::string& s) throw(std::out_of_range);

  /** Convert a long to a string
  */
  static std::string Long2String( long v );

  /** Convert a string to a long
  */
  static long String2Long( const std::string& s) throw(std::out_of_range);

  /** Convert an unsigned long bool to a string
  */
  static std::string ULong2String( unsigned long v );

  /** Convert a string to an unsigned long
  */
  static unsigned long String2ULong( const std::string& s) throw(std::out_of_range);

  /** Convert an unsigned long bool to a string
  */
  static std::string ULLong2String( uint64_t v );

  /** Convert a string to an unsigned uint64_t
  */
  static uint64_t String2ULLong( const std::string& s) throw(std::out_of_range);

  /** Convert a double to a string
  */
  static std::string Double2String( double v);
  
  /** Convert a string to a list of string (delimiter is ',')
  */
  static void String2lstStr( std::list<std::string>& retval, const std::string& s, char delimiter =',');

  /** Convert a string to a double
  */
  static double String2Double( const std::string& s) throw(std::out_of_range);
  //@}
  
  
protected:
  void set_defaultvalue();

  struct ModuleInfo : Module {
    ModuleInfo( const Module& m, Properties* p )
     : Module(m), properties(p)
    {}
    ModuleInfo( const ModuleInfo& mi )
     : Module(mi), properties(mi.properties)
    {}
    ModuleInfo& operator=( const ModuleInfo& mi )
    { Module::operator=( mi ); properties = mi.properties; return *this; }

    ModuleInfo( Properties* p )
     : Module(), properties(p)
    {}
    Properties* properties;
  };
  typedef Module::Options Options;
  typedef std::map<std::string,ModuleInfo*> AllModules;
  AllModules _lookup;
};

} // namespace
#endif // A1_PROPERTY 
