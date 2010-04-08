// =========================================================================
// (c) INRIA, projet MOAIS, 2006
// Author: T. Gautier
// Modif: L. Pigeon
//
// =========================================================================
#include "utils_init.h"
#include "kaapi_dfgdot.h"
#include "kaapi_access.h"
#include "kaapi_gd.h"

namespace DFG {

// -------------------------------------------------------------------------
ODotStream::AccessLink::~AccessLink()
{
  epilogue( *output );
}

// -------------------------------------------------------------------------
void ODotStream::AccessLink::prologue( std::ostream& o ) const
{
  std::ostringstream attribut;
  o << 's' << (unsigned long)access;
  if (Util::KaapiComponentManager::prop["dot.fulldfg"] == "true" )
    o << "[label=\"" << access 
      << "\\ngdid:" << (access->get_gd() ==0 ? Util::ContextId(0) : access->get_gd()->get_id())
      << "\\nisready:" << (access->is_ready() ? 1 : 0)
      << "\\nmode:" << AccessMode::TABMODE[access->get_mode()]
      << "\\nr_version:" << access->get_read_version()
      << "\\nw_version:" << access->get_write_version()
      << "\"";
  else 
    o << "[label=\"\"";

  if (Util::KaapiComponentManager::prop["dot.access.shape"] != "") 
  {
    o << ", shape=" << Util::KaapiComponentManager::prop["dot.access.shape"];
    if (Util::KaapiComponentManager::prop["dot.access.color"] != "")
      o << ", style=filled, color=" << Util::KaapiComponentManager::prop["dot.access.color"];
  }
  o << "];\n";
  
  o << 's' << (unsigned long)access << " -> " << 'c' << (unsigned long)access->get_clo() << "[style=dotted]\n";
  
  switch (mode) {
    case Dir::forward:
      o << 's' << (unsigned long)access << " -> " << 'c' << (unsigned long)closure;
    break;

    case Dir::reverse:
      o << 'c' << (unsigned long)closure << " -> " << 's' << (unsigned long)access;
    break;

    case Dir::reversedot:
      o << 'c' << (unsigned long)closure << " -> " << 's' << (unsigned long)access;
      attribut << "arrowhead=invdot";
    break;

    case Dir::both:
      o << 's' << (unsigned long)access << " -> " << 'c' << (unsigned long)closure;
      attribut << "dir=both";
    break;
  }

  if (Util::KaapiComponentManager::prop["dot.link.access.line.width"] != "1") {
    if (attribut.str() != "")
      attribut << ", ";
    attribut << "style=\"setlinewidth(" << Util::KaapiComponentManager::prop["dot.link.access.line.width"] << ")\"";
  }
  if (Util::KaapiComponentManager::prop["dot.link.access.line.color"] != "default") {
    if (attribut.str() != "")
      attribut << ", ";
    attribut << "color=" << Util::KaapiComponentManager::prop["dot.link.access.line.color"];
  } 
  if (Util::KaapiComponentManager::prop["dot.link.access.constraint"] != "default") {
    if (attribut.str() != "")
      attribut << ", ";
    attribut << "constraint=" << Util::KaapiComponentManager::prop["dot.link.access.constraint"];
  } 
  if (attribut.str() != "")
    o << '[' << attribut.str() << "];\n";
  else
    o << ";\n";
  o << 's' << (unsigned long)access << " -> " << 'c' << (unsigned long)closure << "[style=dotted]\n";

  if (access->get_next() !=0) {
    o << 's' << (unsigned long)access << " -> " << 's' << (unsigned long)access->get_next();
    o << ";\n";
  }
  
  if (Util::KaapiComponentManager::prop["dot.access.gd"] == "true" ) 
  {
    o << "gd" << access->get_gd();
    if (Util::KaapiComponentManager::prop["dot.fulldfg"] == "true" )
    {
      o << "[label=\"" << access->get_gd();
      if (access->get_gd() !=0)
      {
        o  << "\\nid="  << access->get_gd()->get_id()
          << "\\nisstack="  << access->get_gd()->is_instack()
          << "\\nisheap="  << access->get_gd()->is_inheap()
          << "\\nsite="  << access->get_gd()->get_site()
          << "\\n@data=" << access->get_gd()->get_data()
          << "\\nformat=" << access->get_gd()->get_format()->get_name();
        if (access->get_gd()->get_data() !=0)
        {
          std::ostringstream output_data;
          access->get_gd()->get_format()->print( output_data, access->get_gd()->get_data() );
          o << "\\nvalue=<" << output_data.str() << ">";
        }
      }
      o << "\"";
    }
    else 
      o << "[label=\"\"";
    o << ", shape=diamond, style=filled];\n";

    o << "gd" << access->get_gd() << " -> " << 's' << (unsigned long)access << "[style=filled, dir=both]" << std::endl;
  }

}

// -------------------------------------------------------------------------
void ODotStream::AccessLink::epilogue( std::ostream&) const
{
}



// -------------------------------------------------------------------------
ODotStream::SyncLink::~SyncLink()
{
  epilogue( *output );
}

// -------------------------------------------------------------------------
void ODotStream::SyncLink::prologue( std::ostream& o ) const
{
  o << 'c' << (unsigned long)c1 << " -> " << 'c' << (unsigned long)c2 << "[label=\"synclink\"];" << std::endl;
}

// -------------------------------------------------------------------------
void ODotStream::SyncLink::epilogue( std::ostream& ) const
{
}


// -------------------------------------------------------------------------
ODotStream::OriginalLink::~OriginalLink()
{
  epilogue( *output );
}

// -------------------------------------------------------------------------
void ODotStream::OriginalLink::prologue( std::ostream& o ) const
{
  o << 'c' << (unsigned long)c1 << " -> " << 'c' << (unsigned long)c2 << "[label=\"orignal\"];" << std::endl;
}

// -------------------------------------------------------------------------
void ODotStream::OriginalLink::epilogue( std::ostream& ) const
{
}


// -------------------------------------------------------------------------
ODotStream::VersionLink::~VersionLink()
{
  epilogue( *output );
}

// -------------------------------------------------------------------------
void ODotStream::VersionLink::prologue( std::ostream& o ) const
{
  std::ostringstream attribut;
  //TG:SUBCLUSTER o << "\tsubgraph  cluster_" << closure->get_partition() << "{" << std::endl;
 
  o << "gd" << gd << 'v' << version;
  if (Util::KaapiComponentManager::prop["dot.fulldfg"] == "true" )
    o << "[label=\"" << "gd" << gd << 'v' << version
      << "\\nid=" << gd->get_id() << "\"";
  else 
    o << "[label=\"\"";

  if (Util::KaapiComponentManager::prop["dot.version.shape"] != "") 
  {
    o << ", shape=" << Util::KaapiComponentManager::prop["dot.version.shape"];
    if (Util::KaapiComponentManager::prop["dot.version.color"] != "")
      o << ", style=filled, color=" << Util::KaapiComponentManager::prop["dot.version.color"];
  }
  o << "];\n";
  
  switch (mode) {
    case Dir::forward:
      o << "gd" << gd << 'v' << version << " -> " << 'c' << (unsigned long)closure;
      break;

    case Dir::reverse:
      o << 'c' << (unsigned long)closure << " -> " << "gd" << gd << 'v' << version;
      break;

    default:
      break;
  }

  if (Util::KaapiComponentManager::prop["dot.link.version.line.width"] != "1") {
    if (attribut.str() != "")
      attribut << ", ";
    attribut << "style=\"setlinewidth(" << Util::KaapiComponentManager::prop["dot.link.version.line.width"] << ")\"";
  }
  if (Util::KaapiComponentManager::prop["dot.link.version.line.color"] != "default") {
    if (attribut.str() != "")
      attribut << ", ";
    attribut << "color=" << Util::KaapiComponentManager::prop["dot.link.version.line.color"];
  } 
  if (Util::KaapiComponentManager::prop["dot.link.version.constraint"] != "default") {
    if (attribut.str() != "")
      attribut << ", ";
    attribut << "constraint=" << Util::KaapiComponentManager::prop["dot.link.version.constraint"];
  }   
  if (attribut.str() != "")
    o << '[' << attribut.str() << "];\n";
  else
    o << ";\n";

  //TG:SUBCLUSTER o << "}" << std::endl;
}

// -------------------------------------------------------------------------
void ODotStream::VersionLink::epilogue( std::ostream& ) const
{
}



} // namespace Core
