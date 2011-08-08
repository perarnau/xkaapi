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


#include <iostream>
#include "rose_headers.h"
#include "kaapi_task.h"
#include "kaapi_ssa.h"
#include "globals.h"


bool KaapiSSATraversal::is_nested_call(SgNode* node)
{
  node = node->get_parent();
  while (node)
  {
    if (isSgFunctionCallExp(node))
      return true;
    node = node->get_parent();
  }
  return false;
}


bool KaapiSSATraversal::is_returned_call(SgNode* node)
{
  node = node->get_parent();
  while (node)
  {
    if (node->variantT() == V_SgReturnStmt)
      return true;
    node = node->get_parent();
  }
  return false;
}


void KaapiSSATraversal::visit(SgNode* node)
{
  SgFunctionCallExp* const call_expr = isSgFunctionCallExp(node);
  if (call_expr == NULL) return ;

  SgStatement* const call_stmt =
    SageInterface::getEnclosingStatement(call_expr);
  if (call_stmt == NULL) return ;

  if (call_stmt->getAttribute("kaapinotask") != 0) return ;
  if (call_stmt->getAttribute("kaapiwrappercall") != 0) return ;

  SgFunctionDeclaration* const func_decl =
    call_expr->getAssociatedFunctionDeclaration();
  if (func_decl ==0) 
  {
#if CONFIG_ENABLE_DEBUG
    Sg_File_Info* fileInfo = node->get_file_info();
    std::cerr << "****[kaapi_c2c] Warning: function call expression with empty declaration is ignored.\n"
	      << "     In filename '" << fileInfo->get_filename() << "' LINE: " << fileInfo->get_line()
	      << std::endl;
#endif // CONFIG_ENABLE_DEBUG

    /* no declaration (may be incomplete code or expression call (pointer to function) */
    return;
  }
  SgScopeStatement* const scope = SageInterface::getScope(call_expr);
  SgSymbol* symb = func_decl->search_for_symbol_from_symbol_table();
  KaapiTaskAttribute* kta = (KaapiTaskAttribute*)
    symb->getAttribute("kaapitask");
  if (kta && kta->is_signature) kta = NULL;

  // this is only for retvaled calls
  if ((kta == NULL) || (kta->has_retval == false)) return ;

  if (is_nested_call(node) || is_returned_call(node))
  {
    // create a tmp variable
    std::string tmp_name =
      SageInterface::generateUniqueName(node, true);
    SgType* const tmp_type =
      isSgPointerType(kta->retval.type)->get_base_type();
    SgVariableDeclaration* const tmp_decl =
      SageBuilder::buildVariableDeclaration
      (tmp_name, tmp_type, 0, scope);
    SageInterface::prependStatement(tmp_decl, scope);

    // save parent
    SgNode* const prev_parent = node->get_parent();

    // assign call to tmp
    SgExpression* const tmp_expr =
      SageBuilder::buildVarRefExp(tmp_name, scope);
    SgStatement* const assign_stmt =
      SageBuilder::buildAssignStatement(tmp_expr, call_expr);
    assign_stmt->set_file_info(call_stmt->get_file_info());
    SageInterface::insertStatement(call_stmt, assign_stmt);

    // add the stmt to be synced during finalization pass
    synced_stmt_iterator_type synced_pos =
      all_synced_stmts.find(call_stmt);
    if (synced_pos == all_synced_stmts.end())
    {
      // not found, insert
      synced_stmt_pair_type synced_pair;
      synced_pair.first = call_stmt;
      synced_pair.second.scope_ = scope;

      std::pair<synced_stmt_iterator_type, bool> ret;
      ret = all_synced_stmts.insert(synced_pair);
      // assume(ret.second == true);
      synced_pos = ret.first;
    }

    // TODO: when kaapi_sched_sync(varlist) available,
    // add the variable to be synced on to synced_pos->vars

    // replace by tmp
    SgExpression* const parent_expr =	isSgExpression(prev_parent);
    parent_expr->replace_expression(call_expr, tmp_expr);
  }
}
