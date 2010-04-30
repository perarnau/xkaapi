// =========================================================================
// (c) INRIA, projet MOAIS, 2006
// Author: T. Gautier
// Modif: L. Pigeon
//
// =========================================================================
#include "ka_dot_stream.h"
#include "kaapi_impl.h"

namespace ka {

// -------------------------------------------------------------------------
ka::Parser::Module* dot_module = 0;

// -------------------------------------------------------------------------
Dotifiable::~Dotifiable()
{
}


#if 0
#define NB_COLOR 104
// Table of Color 
static const char* tab_Color[NB_COLOR] =
{
/*ca c est rouge*/
   "coral", "crimson", "darksalmon", "deeppink", "hotpink", "indianred", "lightpink", "lightpink", "maroon", "mediumvioletred", "orangered", "palevioletred", "pink", "red", "salmon", "tomato", "violetred",
/*ca c est marron*/
   "beige", "brown", "burlywood", "chocolate", "darkkhaki", "khaki", "peru", "rosybrown", "saddlebrown", "sandybrown", "sienna", "tan",
/*orange*/
   "darkorange", "orange", "orangered",
/*jaune*/
   "darkgoldenrod", "gold", "goldenrod", "greenyellow", "lightgoldenrod", "lightgoldenrodyellow", "lightyellow", "palegoldenrod", "yellow", "yellowgreen",
/*vert*/
   "chartreuse"/*oblige ;)*/, "darkgreen", "darkolivegreen", "darkseagreen", "forestgreen", "green", "greenyellow", "lawngreen", "lightseagreen", "limegreen", "mediumseagreen", "mediumspringgreen", "mintcream", "olivedrab", "palegreen", "seagreen", "springgreen", "yellowgreen",
/*cyan*/
   "aquamarine", "cyan", "darkturquoise", "lightcyan", "mediumaquamarine", "mediumturquoise", "paleturquoise", "turquoise",
/*blue*/
   "aliceblue", "blue", "blueviolet", "cadetblue", "cornflowerblue", "darkslateblue", "deepskyblue", "dodgerblue", "indigo", "lightblue", "lightskyblue", "lightslateblue", "mediumblue", "mediumslateblue", "midnightblue", "navy" /*in the ...*/, "navyblue", "powderblue", "royalblue", "skyblue", "slateblue", "steelblue",
/*magenta*/
   "blueviolet", "darkorchid", "darkviolet", "magenta", "mediumorchid", "mediumpurple", "mediumvioletred", "orchid", "palevioletred", "plum", "purple" /*et deep alors!*/, "violet", "violetred"
};
#else
#define NB_COLOR 16
// Table of Color 
static const char* tab_Color[NB_COLOR] =
{
   "coral",  "orange", "goldenrod", "darkgreen", "darkturquoise", "darkkhaki", "aliceblue", "magenta", "violetred", "yellowgreen", "olivedrab", "turquoise", "navy", "darkorchid", "yellow", "beige"
};
#endif


// --------------------------------------------------------------------------
void ODotStream::terminate_module()
{
  if (dot_module !=0) delete dot_module;
}

ka::Parser::Module* ODotStream::get_module()
{
  if (dot_module !=0) return dot_module;
  dot_module = new Util::Parser::Module;
  dot_module->set_name("dot");
  dot_module->add_option("fulldfg", "false", "True if full information will be dumped");
  dot_module->add_option("representation", "classic", "'classic' or 'alternative'");
  dot_module->add_option("graph.size", "8.5,11.00", "Default graph size, see DOT User's guide");
  dot_module->add_option("task.style", "filled", "Default task'style, see DOT User's guide");
  dot_module->add_option("task.shape", "ellipse", "Default task'style, see DOT User's guide");
  dot_module->add_option("task.color", "state", "Value: 'state'/'mapping'. Set the color of task using the state of the task or the mapping onto partition, Default: 'state'");
  dot_module->add_option("task.color.init", "blue", "Default task'color for task in init state, see DOT User's guide");
  dot_module->add_option("task.color.exec", "red", "Default task'color for task in exec state, see DOT User's guide");
  dot_module->add_option("task.color.wait", "mediumpurple", "Default task'color for task in waiting state, see DOT User's guide");
  dot_module->add_option("task.color.others", "green", "Default task'color for others tasks, see DOT User's guide");
  dot_module->add_option("task.color.steal", "indianred", "Default task'color for task in steal state, see DOT User's guide");
  dot_module->add_option("task.color.term", "grey", "Default task'color for task in term state, see DOT User's guide");
  dot_module->add_option("task.signal.style", "filled", "Default task signal'style, see DOT User's guide");
  dot_module->add_option("task.signal.color", "green", "Default task signal color, see DOT User's guide");
  dot_module->add_option("task.signal.shape", "trapezium", "Default task signal shape, see DOT User's guide");
  dot_module->add_option("access.shape", "box", "Default shape for access, see DOT User's guide");
  dot_module->add_option("access.color", "gray", "Default color for access, see DOT User's guide");
  dot_module->add_option("access.gd", "false", "Connect to a global data object");
  dot_module->add_option("version.shape", "box", "Default shape for version of global data, see DOT User's guide");
  dot_module->add_option("version.color", "gray", "Default color for version of global data, see DOT User's guide");
  dot_module->add_option("link.access.line.color", "default", "Default color for access link, see DOT User's guide");
  dot_module->add_option("link.access.line.width", "1", "Default linewidth for link between closure and access, see DOT User's guide");
  dot_module->add_option("link.access.constraint", "true", "False causes an access edge to be ignored for rank assignment, see DOT User's guide");
  dot_module->add_option("link.version.line.color", "default", "Default color for version link, see DOT User's guide");
  dot_module->add_option("link.version.line.width", "1", "Default linewidth for link between closure and version of global data, see DOT User's guide");
  dot_module->add_option("link.version.constraint", "true", "False causes a version edge to be ignored for rank assignment, see DOT User's guide");
  dot_module->add_option("link.rfo.linewidth", "4", "Default linewidth for link to represent order between closure, see DOT User's guide");
  dot_module->add_option("link.rfo.color", "red", "Default color for rfo link, see DOT User's guide");
  dot_module->add_option("link.rfo.constraint", "true", "False causes a rfo edge to be ignored for rank assignment, see DOT User's guide");
  return dot_module;
}


// -------------------------------------------------------------------------
ODotStream& ODotStream::operator<< (const Object& obj )
{
  obj.prologue( *_output );
  obj.output = _output;
  return *this;
}


// -------------------------------------------------------------------------
ODotStream::Object::~Object()
{
//  epilogue( *output );
}


// -------------------------------------------------------------------------
void ODotStream::Object::prologue( std::ostream& o ) const
{
  o << name;
}
  
// -------------------------------------------------------------------------
void ODotStream::Object::epilogue( std::ostream& ) const
{
}

// -------------------------------------------------------------------------
ODotStream::Graph::~Graph()
{
  epilogue( *output );
}

// -------------------------------------------------------------------------
void ODotStream::Graph::prologue( std::ostream& o ) const
{
  o << "digraph G_";
  Object::prologue( o );
  o << " {" 
    << std::endl
    << "  size=\"" << ka::KaapiComponentManager::prop["dot.graph.size"] << "\";" 
    << std::endl;
}

// -------------------------------------------------------------------------
void ODotStream::Graph::epilogue( std::ostream& o ) const
{
  o << "}\n\n";
}

// -------------------------------------------------------------------------
ODotStream::SubGraph::~SubGraph()
{
  epilogue( *output );
}

// -------------------------------------------------------------------------
void ODotStream::SubGraph::prologue( std::ostream& o ) const
{
  o << "\tsubgraph  cluster_";
  Object::prologue( o );
  o << " { graph [label=\"@thread_" << (unsigned long)address << "\"]" << std::endl;
}

// -------------------------------------------------------------------------
void ODotStream::SubGraph::epilogue( std::ostream& o ) const
{
  o << "}\n";
}

// -------------------------------------------------------------------------
ODotStream::Task::~Task()
{
  epilogue( *output );
}

const char* TABSTATE[] = {
   "INIT",
   "EXEC",
   "STEAL",
   "WAITING",
   "TERM"
};

// -------------------------------------------------------------------------
void ODotStream::Task::prologue( std::ostream& o ) const
{
  const kaapi_format_t* fclo = kaapi_format_resolvebybody( closure->ebody );

  /* display the closure */
  o << 'c' << (unsigned long)closure << "\[label=\"";
  if (Util::KaapiComponentManager::prop["dot.fulldfg"] == "true" ) 
  {
    o << name
      << ",\\n@=" << closure
      << ", #p=" << (int)fclo->count_params
      << ", state=" << TABSTATE[closure->get_state()]
      << ", next=" << --closure;
#if 0
    if(fclo == &Sched::ClosureFormatBroadcastTask::theformat){
      Sched::BroadcastTask* bcast = (Sched::BroadcastTask*)closure;
      o << ", tag=" << bcast->_tag;
    }
    if(fclo == &Sched::ClosureFormatReceiveTask::theformat){
      Sched::ReceiveTask* rcv = (Sched::ReceiveTask*)closure;
      o << ", tag=" << rcv->_tag;
    }
#endif
    o << "\"";

  } else {
    o << "\"";
  }

  {
    if (shape != "default")
      o << ", shape=" << shape;
    else if (Util::KaapiComponentManager::prop["dot.task.shape"] != "")
      o << ", shape=" << Util::KaapiComponentManager::prop["dot.task.shape"];
    if (Util::KaapiComponentManager::prop["dot.task.style"] != "")
      o << ", style=" << Util::KaapiComponentManager::prop["dot.task.style"];
    if (Util::KaapiComponentManager::prop["dot.task.color"] == "state" )
    {
      if ((closure->get_state() == Closure::S_INIT) && Util::KaapiComponentManager::prop["dot.task.color.init"] != "")
        o << ", color=" << Util::KaapiComponentManager::prop["dot.task.color.init"];
      else if ((closure->get_state() == Closure::S_EXEC) && Util::KaapiComponentManager::prop["dot.task.color.exec"] != "")
        o << ", color=" << Util::KaapiComponentManager::prop["dot.task.color.exec"];
      else if ((closure->get_state() == Closure::S_STEAL) && Util::KaapiComponentManager::prop["dot.task.color.steal"] != "")
        o << ", color=" << Util::KaapiComponentManager::prop["dot.task.color.steal"];
      else if ((closure->get_state() == Closure::S_TERM) && Util::KaapiComponentManager::prop["dot.task.color.term"] != "")
        o << ", color=" << Util::KaapiComponentManager::prop["dot.task.color.term"];
      else if ((closure->get_state() == Closure::S_WAITING) && Util::KaapiComponentManager::prop["dot.task.color.wait"] != "")
        o << ", color=" << Util::KaapiComponentManager::prop["dot.task.color.wait"];
      else
        o << ", color=" << Util::KaapiComponentManager::prop["dot.task.color.others"];
    }
    else if (Util::KaapiComponentManager::prop["dot.task.color"] == "mapping" )
    {
      if ((closure->get_state() == Closure::S_INIT) && Util::KaapiComponentManager::prop["dot.task.color.init"] != "")
        o << ", color=" << tab_Color[closure->get_partition() % (NB_COLOR)];
      else if ((closure->get_state() == Closure::S_EXEC) && Util::KaapiComponentManager::prop["dot.task.color.exec"] != "")
        o << ", color=" << Util::KaapiComponentManager::prop["dot.task.color.exec"];
      else if ((closure->get_state() == Closure::S_STEAL) && Util::KaapiComponentManager::prop["dot.task.color.steal"] != "")
        o << ", color=" << Util::KaapiComponentManager::prop["dot.task.color.steal"];
      else if ((closure->get_state() == Closure::S_TERM) && Util::KaapiComponentManager::prop["dot.task.color.term"] != "")
        o << ", color=" << Util::KaapiComponentManager::prop["dot.task.color.term"];
      else if ((closure->get_state() == Closure::S_WAITING) && Util::KaapiComponentManager::prop["dot.task.color.wait"] != "")
        o << ", color=" << Util::KaapiComponentManager::prop["dot.task.color.wait"];
      else
        o << ", color=" << Util::KaapiComponentManager::prop["dot.task.color.others"];
    }
  }
  o << "];\n" << std::flush;
}

// -------------------------------------------------------------------------
void ODotStream::Task::epilogue( std::ostream& ) const
{
}

// -------------------------------------------------------------------------
ODotStream::OrderLink::~OrderLink()
{
  epilogue( *output );
}

// -------------------------------------------------------------------------
void ODotStream::OrderLink::prologue( std::ostream& o ) const
{
  std::ostringstream attribut;
  if (Util::KaapiComponentManager::prop["dot.link.rfo.linewidth"] != "0") {
    o << 'c' << (unsigned long)closure1 << " -> " << 'c' << (unsigned long)closure2;
    if (Util::KaapiComponentManager::prop["dot.link.rfo.linewidth"] != "1") {
      if (attribut.str() != "")
        attribut << ", ";
      attribut << "style=\"setlinewidth(" << Util::KaapiComponentManager::prop["dot.link.rfo.linewidth"] << ")\"";
    }
    if (Util::KaapiComponentManager::prop["dot.link.rfo.color"] != "default")
    {
      if (attribut.str() != "")
        attribut << ", ";
      attribut << "color=" << Util::KaapiComponentManager::prop["dot.link.rfo.color"];
    } 
    if (Util::KaapiComponentManager::prop["dot.link.rfo.constraint"] != "default") {
      if (attribut.str() != "")
        attribut << ", ";
      attribut << "constraint=" << Util::KaapiComponentManager::prop["dot.link.rfo.constraint"];
    } 
    if (attribut.str() != "")
    {
      o << '[' << attribut.str() << ']';
    }
    o << ";\n";
  }
}

// -------------------------------------------------------------------------
void ODotStream::OrderLink::epilogue( std::ostream& ) const
{
}

// -------------------------------------------------------------------------
ODotStream::Text::~Text()
{
  epilogue( *output );
}

// -------------------------------------------------------------------------
void ODotStream::Text::prologue( std::ostream& o ) const
{ ODotStream::Object::prologue(o); }


// -------------------------------------------------------------------------
void ODotStream::Text::epilogue( std::ostream& ) const
{ }

} // namespace Core
