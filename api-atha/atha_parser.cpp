// ==========================================================================
// (c) INRIA, projet MOAIS, 2006
// Author: Thierry Gautier
// Status: ok
// ==========================================================================
#include <iostream>
#include "atha_parser.h"

namespace atha {

// --------------------------------------------------------------------
Parser::Module::Module()
 : name(""), helper(0)
{}


// --------------------------------------------------------------------
Parser::Module::Module(const std::string& n )
 : name(n), helper(0)
{}


// --------------------------------------------------------------------
const std::string& Parser::Module::get_name() const
{ return name; }


// --------------------------------------------------------------------
void Parser::Module::set_name(const std::string& n )
{ name = n; }


// --------------------------------------------------------------------
void Parser::Module::set_helper( void (*h)(std::ostream& o) )
{  helper = h; }


// --------------------------------------------------------------------
void Parser::Module::add_option( const std::string& name, 
                                 const std::string& value, 
                                 const std::string& usage,
                                 char mode,
                                const std::string& separator
  )
{
  std::string no = "-" + name;
  options.insert(std::make_pair(no,OptionInfo(name,value,usage,mode,separator)));
}


// --------------------------------------------------------------------
std::string Parser::Module::Options::get_canonical_name(const std::string& option_name)
{
  std::string retval = option_name;
  for (unsigned int i=0; i<retval.size(); ++i)
    retval[i] = (retval[i] == '_' ? '.' : retval[i]);
  return retval;
}


// --------------------------------------------------------------------
std::string Parser::Module::get_resource_name(const std::string& option_name) const
{
  return name + "." + Options::get_canonical_name(option_name);
}


// --------------------------------------------------------------------
void Parser::Module::print_usage( std::ostream& o ) const
{
  o << "Usage for module '" << name << "'" << std::endl;
  Options::const_iterator i_beg = options.begin();
  Options::const_iterator i_end = options.end();
  while (i_beg != i_end)
  {
    o << "*** " << i_beg->second.name << std::endl;
    o << "\tCommand line  : " << i_beg->first << std::endl;
    o << "\tResource name : " << get_resource_name(i_beg->second.name) << std::endl;
    o << "\tInsertion mode: " << i_beg->second.mode << std::endl;
    if (i_beg->second.default_value == "<no value>")
      o << "\tDefault value : No value" << std::endl;
    else
      o << "\tDefault value : " << i_beg->second.default_value << std::endl;
    o << "\tUsage         : " << i_beg->second.usage << std::endl;
    ++i_beg;
  }

  if (helper !=0) 
  {
    (*helper)(o << "Extra information:" << std::endl);
  }
}


// --------------------------------------------------------------------
void Parser::Module::dump_rc( std::ostream& o ) const
{
  o << "\n#Dump of predefined property values for module:" << name << std::endl;
  Options::const_iterator i_beg = options.begin();
  Options::const_iterator i_end = options.end();
  while (i_beg != i_end)
  {
    if (i_beg->second.default_value != "<no value>")
    {
      o << get_resource_name(i_beg->second.name) << " = "
        << i_beg->second.default_value
        << ";" << std::endl;
    }
    else {
      o << '#' << get_resource_name(i_beg->second.name) << " = "
        << "<NO DEFAULT VALUE, see documentation>;" << std::endl;
    }
    ++i_beg;
  }
}


// --------------------------------------------------------------------
Parser::Parser()
{
  _lookup.clear();
}


// --------------------------------------------------------------------
Parser::~Parser()
{
  AllModules::iterator ibeg = _lookup.begin();
  AllModules::iterator iend = _lookup.end();
  while (ibeg != iend)
  {
    delete ibeg->second;
    ++ibeg;
  }

  _lookup.clear();

}


// --------------------------------------------------------------------
void Parser::parse( Properties& initialprop, int& argc, char**& argv )
{
  enum StateParser {
    S_BEGIN,
    S_MODULE_FOUND,
    S_OPTION_FOUND,
    S_END_OPTION
  };
  StateParser state = S_BEGIN;

  AllModules::iterator ismod;
//  Options::iterator isopt;
  ModuleInfo* module_info = 0;
  std::string module_name;
  std::string option_name;
  std::string key;
  char mode_option =0;
  std::string separator;

  /* initialize all options that have default value */
  set_defaultvalue();
  AllModules::iterator ibeg = _lookup.begin();
  AllModules::iterator iend = _lookup.end();
  while (ibeg != iend)
  {
    ibeg->second->properties->merge(initialprop);
    ++ibeg;
  }
  
  if ((argc ==0) || (argv ==0)) return;

  /* Define ALLOW_MIX_OPTIONS if you want to be able to mix kaapi
   * options with program options/arguments else, kaapi options MUST
   * precede program options/arguments In any case, argument
   * processing is stopped by "--" (which is removed for the
   * application.
   */
#define ALLOW_MIX_OPTIONS

  /* To help debugging the following code.
   * Can be removed once well tested
   */
  //#define DEBUG_PARSING_OPTIONS

#ifdef DEBUG_PARSING_OPTIONS
#define ADD_VALUE(v) \
    std::cout << "adding value " << v << std::endl;
#else
#define ADD_VALUE(v)
#endif

#define WRITE_VALUE(val) ({ \
      std::string value=val;			   \
      ADD_VALUE(value) \
      if (value == "<no value>") value = "true";   \
      else if (mode_option == 'w')		   \
	(*module_info->properties)[key] = value;		  \
      else if ((*module_info->properties)[key] == "<no value>")	  \
	(*module_info->properties)[key] = value;		  \
      else							  \
	(*module_info->properties)[key] += separator + value;	  \
    })

  /* look for module name and recopy non recognized options */
  //int i;
  for (int i=1; i< argc; ++i)
  {
    std::string argv_curr((argv)[i]);

#ifdef DEBUG_PARSING_OPTIONS    
    std::cout << "managing " << argv_curr << std::endl;
#endif

    // entering help section
    if (argv_curr == "--help"|| argv_curr == "-h" ) {

      // look for module name after help command
      if(i+1< argc ) 
      {
        module_name = std::string( (argv)[i+1]);
        ismod = _lookup.find( std::string("--")+module_name);

        // try to recognize the module name
        if  (ismod != _lookup.end()) {
          module_info = ismod->second;
          option_name ="";
          mode_option ='w';
          separator = "";
          module_info->print_usage(std::cout);
          exit(EXIT_SUCCESS);
        }
        else {
          std::cerr << "module " << module_name << "is unknown" << std::endl;
        }
      }

      // display generic help
      this->print_usage(std::cout);
      exit(EXIT_SUCCESS);

    }

    else if (argv_curr == "--dumprc")
    {
      std::ofstream file("dump.rc");
      this->dump_rc(file);
      exit(EXIT_SUCCESS);
    }

    /* very special options have been managed. Let's go with normal options
     */

    /* end of argument parsing ? */
    if (argv_curr == "--")
    {
      if (state == S_OPTION_FOUND) {
          WRITE_VALUE("<no value>");
      }
      state = S_BEGIN;
      /* skip this argument: the application should not see it */
      argv[i++]=0;

#ifdef DEBUG_PARSING_OPTIONS    
      std::cout << " => stopping parsing args on --" << std::endl;
#endif
      break;
    }
    else
    {

        /* Is it a module options ? */
        /* find definition of the options for the module argv[i] */
        ismod = _lookup.find( argv[i] );
        if  (ismod != _lookup.end())
        {
            /* It is a module: we record it */
            module_name = std::string( (argv)[i]+2 );
            module_info = ismod->second;
            /* If a option without value has been previously found, we write
               the default value */
            if (state == S_OPTION_FOUND) {       
                WRITE_VALUE("<no value>");
            }
            state = S_MODULE_FOUND;
#ifdef DEBUG_PARSING_OPTIONS    
            std::cout << " => found module " << argv_curr << std::endl;
#endif
            argv[i]=0;
            continue;
        }
        else
        {

            /* options must be introduce by a module */
            if (state==S_BEGIN)
            {
#ifdef ALLOW_MIX_OPTIONS
#  ifdef DEBUG_PARSING_OPTIONS    
                std::cout << "skiping argv[" << i << "] = "  << argv[i] << std::endl;
#  endif
                continue;
#else
                break;
#endif
            }

            /* look for option name */
            Options::iterator isopt = module_info->options.find(argv[i]);
            if (isopt != module_info->options.end() ) 
            {
#ifdef DEBUG_PARSING_OPTIONS    
                std::cout << "  => found option " << argv_curr << std::endl;
#endif
                if (state == S_OPTION_FOUND) {
                    WRITE_VALUE("<no value>");
                }
                state = S_OPTION_FOUND;
                option_name = std::string(argv[i]+1);
                mode_option = isopt->second.mode;
                separator = isopt->second.separator;
                //key = module_name + "." + Options::get_canonical_name(option_name);
                key = module_info->get_resource_name(option_name);
                argv[i]=0;
                continue;
            }
            else if (state == S_OPTION_FOUND)
            {
#ifdef DEBUG_PARSING_OPTIONS    
                std::cout << "  => found value " << argv_curr << std::endl;
#endif
                WRITE_VALUE(argv[i]);
                argv[i]=0;
                state = S_MODULE_FOUND;
            }
            else
            {
#ifdef ALLOW_MIX_OPTIONS
#ifdef DEBUG_PARSING_OPTIONS    
                std::cout << "skiping argv[" << i << "] = " << argv[i] << std::endl;
#endif
                continue;
#else
                break;
#endif
            }
        }
    }
  } /* for */
  
  if (state == S_OPTION_FOUND) 
  {
    /* write previous option */
    WRITE_VALUE("<no value>");
    state = S_BEGIN;
  }

  //remove useless NULL args
  int j= 1;
  for(int k=1;j<argc;++j,++k)
  {
      while(k<argc && argv[k] == 0) { ++k; } ;
      if(k==argc) break;
      else if (k > j ) { argv[j]=argv[k]; argv[k]=0; }
  }
  argc=j;

}


// --------------------------------------------------------------------
void Parser::set_defaultvalue()
{
  AllModules::const_iterator beg_mod = _lookup.begin();
  AllModules::const_iterator end_mod = _lookup.end();
  while (beg_mod != end_mod)
  {
    ModuleInfo* module_info = beg_mod->second;
    std::string module_name = beg_mod->first.substr(2);

    /* add default value for each options  */
    Options::iterator beg_opt = module_info->options.begin();
    Options::iterator end_opt = module_info->options.end();
    while (beg_opt != end_opt)
    {
      if (true)//beg_opt->second.default_value != "<no value>") 
      {
        std::string option_name = beg_opt->first.substr(1);
        std::string key = module_info->get_resource_name(option_name);
        if ((*module_info->properties)[ key ] == "")
          (*module_info->properties)[ key ] = beg_opt->second.default_value;
      }
      ++beg_opt;
    }
    ++beg_mod;
  }
}


// --------------------------------------------------------------------
void Parser::add_parser(const Parser& p)
{
  AllModules::const_iterator ibeg = p._lookup.begin();
  AllModules::const_iterator iend = p._lookup.end();
  while (ibeg != iend)
  {
    /* add the module */
    AllModules::iterator curr = _lookup.find(ibeg->first); 
    ModuleInfo* mi_src = ibeg->second;
    ModuleInfo* mi_dest = 0;
    if (curr == _lookup.end())
      _lookup.insert( std::make_pair( ibeg->first, mi_dest = new ModuleInfo( mi_src->properties) ) );
    else
      mi_dest = curr->second;
    
    /* add all options from mi_src */
    Options::iterator beg_opt = mi_src->options.begin();
    Options::iterator end_opt = mi_src->options.end();
    while (beg_opt != end_opt)
    {
      mi_dest->options.insert( *beg_opt );
      ++beg_opt;
    }
  }
}


// --------------------------------------------------------------------
void Parser::add_module(const Parser::Module* mod, Properties* prop)
    throw(InvalidArgumentError)
{
  if (mod == 0) return;
  if (prop == 0) return;
  std::string m = mod->get_name();
  if (m == "")
    Exception_throw( InvalidArgumentError("[Parser::add_module] module has no name") );
  std::string nm = "--" + m;
  AllModules::iterator curr = _lookup.find(nm);
  ModuleInfo* mi =0;
  if (curr != _lookup.end())
    Exception_throw( InvalidArgumentError("[Parser::add_module] module already exist") );
  _lookup.insert( std::make_pair( nm, mi = new ModuleInfo(*mod,prop) ) );
}

// --------------------------------------------------------------------
void Parser::add_module(const std::string& m, Properties* prop )
    throw(InvalidArgumentError)
{
  if (prop == 0) return;
  std::string nm = "--" + m;
  AllModules::iterator curr = _lookup.find(nm);
  ModuleInfo* mi =0;
  if (curr != _lookup.end())
    Exception_throw( InvalidArgumentError("[Parser::add_module] module already exist") );
  _lookup.insert( std::make_pair( nm, mi = new ModuleInfo(prop) ) );
}


// --------------------------------------------------------------------
void Parser::print_usage( std::ostream& o ) const
{
  o << "General options:" << std::endl
    << "\t --help|-h : display this help and exit" << std::endl
    << "\t --help <module name> : display help for a module" << std::endl
    << "\t --dumprc  : dump in the file 'dump.rc' the predefined values of the properties for each module" << std::endl
    << "Note on passing option in command line." << std::endl
    << "\t KAAPI library is built by several software component, each component as a name" << std::endl
    << "\t display in the following list. To set an option for a component, the user may" << std::endl
    << "\t set it into the kaapi.rc file or pass it by command line argument. The format is" << std::endl
    << "\t --<module name> -<option name1> [<value1>] [-<option name2> [<value2>]]." << std::endl
    << "Module list:" << std::endl;

  AllModules::const_iterator ibeg = _lookup.begin();
  AllModules::const_iterator iend = _lookup.end();
  while (ibeg != iend)
  {
    o << ibeg->second->get_name() << std::endl;
    ++ibeg;
  }
}


// --------------------------------------------------------------------
void Parser::dump_rc( std::ostream& o ) const
{
  AllModules::const_iterator ibeg = _lookup.begin();
  AllModules::const_iterator iend = _lookup.end();
  while (ibeg != iend)
  {
    ibeg->second->dump_rc( o );
    ++ibeg;
  }
}

// --------------------------------------------------------------------
const std::string& Parser::Bool2String( bool v )
{
  static std::string s_true = "true";
  static std::string s_false= "false";
  if (v) return s_true;
  else return s_false;
}

bool Parser::String2Bool( const std::string& s) throw(InvalidArgumentError)
{
  if (s == "true") return true;
  if (s == "false") return false;
  Exception_throw( InvalidArgumentError("[String2Bool]") );
  return false;
}

// --------------------------------------------------------------------
void Parser::String2lstStr( std::list<std::string>& lst, const std::string& s, char delimiter )
{
  std::string tmp = "";
  for(unsigned int i = 0; i < s.length(); ++i)
  {
    if (s[i] == delimiter)
    {
        if(tmp.length() != 0)
          lst.push_back(tmp);
        tmp = "";
    }
    else if (s[i] == ' ') {
    }
    else tmp.push_back(s[i]);
  }
  if(tmp.length() != 0)
    lst.push_back(tmp);

}

// --------------------------------------------------------------------
long Parser::String2Long( const std::string& s) throw(InvalidArgumentError)
{
  long v;
  std::istringstream tmp(s); 
  tmp >> std::ws >> v;
  return v;
}

std::string Parser::Long2String( long v )
{
  std::ostringstream tmp; tmp << v;
  return tmp.str();
}

// --------------------------------------------------------------------
unsigned long Parser::String2ULong( const std::string& s) throw(InvalidArgumentError)
{
  unsigned long v;
  std::istringstream tmp(s); 
  tmp >> std::ws >> v;
  return v;
}

std::string Parser::ULong2String( unsigned long v )
{
  std::ostringstream tmp; tmp << v;
  return tmp.str();
}


// --------------------------------------------------------------------
unsigned long long Parser::String2ULLong( const std::string& s) throw(InvalidArgumentError)
{
  unsigned long long v;
  std::istringstream tmp(s); 
  tmp >> std::ws >> v;
  return v;
}

std::string Parser::ULLong2String( unsigned long long v )
{
  std::ostringstream tmp; tmp << v;
  return tmp.str();
}


// --------------------------------------------------------------------
double Parser::String2Double( const std::string& s) throw(InvalidArgumentError)
{
  double v;
  std::istringstream tmp(s); 
  tmp >> std::ws >> v;
  return v;
}

std::string Parser::Double2String( double v)
{
  std::ostringstream tmp; tmp << v;
  return tmp.str();
}

} //namespace
