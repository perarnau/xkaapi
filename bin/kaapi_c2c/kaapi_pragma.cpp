/*
** xkaapi
** 
** Created on Tue Mar 31 15:19:09 2009
** Copyright 2009 INRIA.
**
** Contributors :
**
** thierry.gautier@inrialpes.fr
** fabien.lementec@gmail.com / fabien.lementec@imag.fr
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


#include <string>
#include "rose_headers.h"
#include "kaapi_pragma.h"
#include "parser.h"


void KaapiPragma::normalizePragmaDeclaration(SgPragmaDeclaration* sgp)
{
  // building a pragma from the statement corrects the multiline related bug

  SgPragma* const old_pragma = sgp->get_pragma();
  std::string decl_string = old_pragma->get_pragma();

  std::string::iterator pos = decl_string.begin();
  std::string::iterator end = decl_string.end();
  for (; pos != end; ++pos) if (*pos == '\n') *pos = ' ';

  SgPragma* const normalized_pragma =
    new SgPragma(decl_string, old_pragma->get_file_info());
  sgp->set_pragma(normalized_pragma);
  delete old_pragma;
}


KaapiPragma::KaapiPragma()
{
}


void KaapiPragma::operator()(SgNode* node)
{
  if (node->variantT() == V_SgPragmaDeclaration) 
  {
    Sg_File_Info* fileInfo = node->get_file_info();
    SgPragmaDeclaration* sgp = isSgPragmaDeclaration(node);

    normalizePragmaDeclaration(sgp);
      
    // Parse if it is a Kaapi Pragma
    std::string pragma_string = sgp->get_pragma()->get_pragma();
    Parser parser( pragma_string.c_str() );

    std::string name;
    parser.ParseIdentifier( name );
      
    /* css compatibility */
    if (name == "css") name = "kaapi";
    if (name != "kaapi") return;
      
    parser.ParseIdentifier( name );

    /* Parallel kaapi region
       add begin_parallel_region and end_parallel_region between the basic block
    */
    if (name == "parallel")
    {
      parser.DoKaapiPragmaParallelRegion( sgp );
    } 

    /* Case of task definition: because do no processing but register what to do
       then also pass the string in order the next step will be able to do parsing.
    */
    else if (name == "task")
    {
      parser.DoKaapiPragmaTask( sgp );
    } /* end for task */

    else if (name == "signature" || name == "prototype")
    {
      parser.DoKaapiPragmaSignature( sgp );
    }

    else if (name == "notask")
    {
      parser.DoKaapiPragmaNoTask( sgp );
    } 
       
    /* Case of barrier definition
     */
    else if ((name == "barrier") || (name == "sync")) // css compability
    {
      parser.DoKaapiPragmaBarrier( sgp );
    }

    /* Case of waiton clause
     */
    else if (name == "waiton") 
    {
      parser.DoKaapiPragmaWaiton( sgp );
    }

    /* Case of variable scope
     */
    else if (name == "data") 
    {
      parser.DoKaapiPragmaData( node );
    }

    else if (name == "loop")
    {
      parser.DoKaapiPragmaLoop( sgp );
      return;
    }

    /* Case of init/finalize
     */
    else if ((name == "init") || (name == "start"))
    {
      parser.DoKaapiPragmaInit( sgp, true );
    }
    else if (name == "finish") 
    {
      parser.DoKaapiPragmaInit( sgp, false );
    }
    else if (name == "declare") 
    {
      parser.DoKaapiPragmaDeclare( sgp );
    }
    else if (name == "mutex") 
    {
      std::cerr << "****[kaapi_c2c] Warning: #pragma mutex is ignored.\n"
		<< "     In filename '" << fileInfo->get_filename() << "' LINE: " << fileInfo->get_line()
		<< std::endl;
    }
    else
    {
      std::cerr <<  "In filename '" << fileInfo->get_filename() << "' LINE: " << fileInfo->get_line()
		<< " unknown #pragma kaapi " << name
		<< std::endl;
      KaapiAbort("*** Error");
    }
  } // variantT == V_SgPragmaDeclaration
}  
