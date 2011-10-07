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


#include <stdio.h>
#include <set>
#include <vector>
#include "rose_headers.h"
#include "kaapi_mode_analysis.h"
#include "kaapi_c2c_task.h"
#include "globals.h"


__attribute__((unused))
static SgNode* getDeclByInitializedName
(SgFunctionDeclaration* func_decl, SgInitializedName* name)
{
  SgSymbol* const symbol = name->get_symbol_from_symbol_table();
  if (symbol == NULL) return NULL;
  return isSgVariableSymbol(symbol)->get_declaration();
}

static inline bool isWriteBinaryOp(SgBinaryOp* op)
{
  switch (op->variantT())
  {
  case V_SgAssignOp:
  case V_SgPlusAssignOp:
  case V_SgMinusAssignOp:
  case V_SgAndAssignOp:
  case V_SgIorAssignOp:
  case V_SgMultAssignOp:
  case V_SgDivAssignOp:
  case V_SgModAssignOp:
  case V_SgXorAssignOp:
  case V_SgLshiftAssignOp:
  case V_SgRshiftAssignOp:
  case V_SgExponentiationOp:
  case V_SgConcatenationOp:
  case V_SgPointerAssignOp:
    return true;
    break ;

  default: break;
  }

  return false;
}

static inline bool isWriteUnaryOp(SgUnaryOp* op)
{
  switch (op->variantT())
  {
  case V_SgPlusPlusOp:
  case V_SgMinusMinusOp:
    return true;
    break ;

  default: break ;
  }

  return false;
}

static inline bool isRhsOperand(SgBinaryOp* op, SgNode* node)
{
  return op->get_rhs_operand() == node;
}

static inline bool isDerefExpr(SgNode* node, SgNode* child)
{
  if (isSgPointerDerefExp(node)) return true;
  else if (isSgPntrArrRefExp(node))
    return (isRhsOperand(isSgBinaryOp(node), child) == false);
  return false;
}

static inline bool mayHaveSideEffects
(
 SgProject* project,
 std::set<SgFunctionDeclaration*>& explored_decls,
 SgNode* node,
 SgNode* child
)
{
  SgFunctionCallExp* const call_expr = isSgFunctionCallExp(node);
  if (call_expr == NULL) return false;

  SgExpressionPtrList const args_expr =
    call_expr->get_args()->get_expressions();

  // find the argument pos, then its parameter name
  SgExpressionPtrList::const_iterator pos = args_expr.begin();
  SgExpressionPtrList::const_iterator end = args_expr.end();
  unsigned int i = 0;
  for (; pos != end; ++pos, ++i)
    if (isSgNode(*pos) == child)
      break ;

  // not found
  if (pos == end) return false;

  // find the function parameter list
  SgExpression* const func_expr = call_expr->get_function();

  // tutorial/resolveOverloadedFunction.C
  SgFunctionRefExp* const func_ref = isSgFunctionRefExp(func_expr);
  SgFunctionSymbol* func_sym = NULL;
  if (func_ref != NULL)
  {
    // non member function
    func_sym = func_ref->get_symbol();
  }
  else
  {
    SgDotExp* const dot_expr = isSgDotExp(func_expr);
    // assume(dot_expr);
    func_sym = isSgMemberFunctionRefExp
      (dot_expr->get_rhs_operand())->get_symbol();
  }
  SgFunctionDeclaration* const func_decl = func_sym->get_declaration();
  // assume(func_decl);
  SgFunctionParameterList* const func_params = func_decl->get_parameterList();
  // assume(func_params);
  SgInitializedNamePtrList& name_list = func_params->get_args();

  // get the ith argument name
  std::vector<SgInitializedName*> param_names;
  param_names.resize(1);
  param_names[0] = name_list[i];

  // run the mode analysis
  const bool is_success = DoKaapiModeAnalysis
    (project, explored_decls, func_decl, param_names);
  return is_success ? false : true;
}

static bool isWriteVarRefExp
(
 SgProject* project,
 std::set<SgFunctionDeclaration*>& explored_decls,
 SgVarRefExp* ref_expr
)
{
  // TODO: explored_decls is not enough. there should
  // be the parameter position to handle calls to the
  // same function but with different args position.
  // ie. foo(bar, baz) is different from foo(baz, bar)

  // a more correct algorithm should consider:
  // . enclosing expression: if we gets out of this
  // expression, stop the analysis: either the statement
  // is reached (current case), or we get out of the
  // enclosing block (ie. arr[n] where [n] the scope)

  SgNode* child = isSgNode(ref_expr);
  SgNode* node = child->get_parent();

  // last expression seen
  SgNode* last_expr = NULL;

  // deref_level == 1 means a value access
  int deref_level = 0;

  while (1)
  {
    // stop on the first binop
    // or on the first statement

    if (node == NULL) break ;
    else if (isSgStatement(node)) break ;

    // needed for tracking call args
    if (isSgExprListExp(child) == NULL)
      last_expr = child;

    // check first for deref expression
    // sgPntrArrayRefExp is a binop too
    if (isDerefExpr(node, child) == true)
    {
      ++deref_level;
    }
    // check for underef expr
    else if (isSgAddressOfOp(node))
    {
      --deref_level;
    }
    else if (deref_level == 1) // check for modifying op
    {
      SgBinaryOp* const binop = isSgBinaryOp(node);
      SgUnaryOp* const unop = isSgUnaryOp(node);

      if ((binop != NULL) || (unop != NULL))
      {
	if (binop != NULL)
	{
	  if (isRhsOperand(binop, child)) return false;
	  return isWriteBinaryOp(binop);
	}
	else // unop != NULL
	{
	  return isWriteUnaryOp(unop);
	}
      }
    }
    else if (deref_level <= 0)
    {
      if (mayHaveSideEffects(project, explored_decls, node, last_expr))
	return true;
      // continue otherwise
    }

    // one level up
    child = node;
    node = node->get_parent();
  }

  return false;
}

__attribute__((unused))
static SgStatement* getStatement(SgNode* node)
{
  while (node)
  {
    if (isSgStatement(node)) return isSgStatement(node);
    node = node->get_parent();
  }

  return NULL;
}

__attribute__((unused))
static inline SgStatement* getStatement(SgVarRefExp* ref_expr)
{
  return getStatement(isSgNode(ref_expr));
}

__attribute__((unused))
static inline SgStatement* getStatement(SgExpression* expr)
{
  return getStatement(isSgNode(expr));
}


// exported

bool DoKaapiModeAnalysis
(
 SgProject* project,
 std::set<SgFunctionDeclaration*>& explored_decls,
 SgFunctionDeclaration* func_decl,
 std::vector<SgInitializedName*>& param_names
)
{
  // param_names the names of the parameters
  // which we want detect write access on

  // already explored
  if (explored_decls.find(func_decl) != explored_decls.end())
    return true;
  explored_decls.insert(func_decl);

  // possible for a decl not to have a definition
  SgFunctionDefinition* const func_def = func_decl->get_definition();
  if (func_def == NULL)
  {
    // run analysis on the task parameters

    SgSymbol* func_sym;
    KaapiTaskAttribute* kta;
    std::vector<SgInitializedName*>::iterator name_pos, name_end;
    std::vector<KaapiTaskFormalParam>::iterator param_pos, param_end;
    enum { fu, bar, baz } reason = fu;

    func_sym = func_decl->search_for_symbol_from_symbol_table();
    if (func_sym == NULL) goto on_analysis_failure;

    kta = (KaapiTaskAttribute*)func_sym->getAttribute("kaapitask");
    if (kta == NULL) goto on_analysis_failure;

    // foreach param_name, check it is a readonly mode
    name_pos = param_names.begin();
    name_end = param_names.end();
    for (; name_pos != name_end; ++name_pos)
    {
      // find the corresponding task attribute param
      param_pos = kta->formal_param.begin();
      param_end = kta->formal_param.end();
      for (; param_pos != param_end; ++param_pos)
	if ((*name_pos)->get_name() == param_pos->initname->get_name())
	  break ;

      // not found, failure
      if (param_pos == param_end)
      {
	reason = bar;
	goto on_analysis_failure;
      }

      // found, is it a readonly mode
      if (param_pos->mode != KAAPI_R_MODE)
      {
	reason = baz;
	goto on_analysis_failure;
      }
    }

    // covered all parameters with success
    return true;

  on_analysis_failure:
    printf("[ MODE ANALYSIS ]\n");
    printf("  ABORTED DUE TO ");
    switch (reason)
    {
    case fu:
      printf("MISSING DEFINITION %s\n", func_decl->get_name().str());
      break ;
    case bar:
      printf("PARAMETER NOT FOUND\n");
      break ;
    case baz:
      printf("INVALID MODE\n");
      break ;
    default: break ;
    }
    return false;
  }

  DefUseAnalysis project_dfa(project);
  DefUseAnalysisPF func_dfa(false, &project_dfa);
  bool abortme = false;
  func_dfa.run(func_def, abortme);

  std::vector<SgInitializedName*>::iterator name_pos = param_names.begin();
  std::vector<SgInitializedName*>::iterator name_end = param_names.end();
  for (; name_pos != name_end; ++name_pos)
  {
    SgInitializedName* const param_name = *name_pos;

    typedef std::vector<std::pair<SgInitializedName*, SgNode*> > multitype;
    std::map<SgNode*, multitype> use_map = project_dfa.getUseMap();
    std::map<SgNode*, multitype>::iterator use_pos = use_map.begin();
    std::map<SgNode*, multitype>::iterator use_end = use_map.end();
    for (; use_pos != use_end; ++use_pos)
    {
      // from DefUseAnalysis.cpp
      // fixme: multiple pass done on the same expression
      // fixme: use node comparison instead of string 

      SgVarRefExp* const ref_expr = isSgVarRefExp(use_pos->first);
      if (ref_expr == NULL) continue ;

      SgInitializedName* const ref_name =
	ref_expr->get_symbol()->get_declaration();
      // assume(ref_name);

      if (ref_name->get_qualified_name() != param_name->get_qualified_name())
	continue ;

      if (isWriteVarRefExp(project, explored_decls, ref_expr))
      {
	printf("[ MODE ANALYSIS WARNING ]\n");
	printf("  INVALID WRITE TO VARIABLE %s\n",
	       param_name->get_qualified_name().str());
	printf("  STATEMENT: \"%s\"\n",
	       getStatement(ref_expr)->unparseToString().c_str());
      }
    }
  }

  return true;
}

bool DoKaapiModeAnalysis(SgProject* project, KaapiTaskAttribute* kta)
{
  // do mode analysis on a given task

  // build a list of param names and analyze

  std::set<SgFunctionDeclaration*> explored_decls;

  std::vector<SgInitializedName*> param_names;

  std::vector<KaapiTaskFormalParam>::iterator
    param_pos = kta->formal_param.begin();
  std::vector<KaapiTaskFormalParam>::iterator
    param_end = kta->formal_param.end();
  for (; param_pos != param_end; ++param_pos)
  {
    // analyze only if readonly
    if (param_pos->mode != KAAPI_R_MODE) continue ;
    param_names.push_back(param_pos->initname);
  }

  return DoKaapiModeAnalysis
    (project, explored_decls, kta->func_decl, param_names);
}

bool DoKaapiModeAnalysis(SgProject* project)
{
  // write on read
  bool is_success = true;

  // check modes of each access
  ListTaskFunctionDeclaration::iterator decl_pos = all_task_func_decl.begin();
  ListTaskFunctionDeclaration::iterator decl_end = all_task_func_decl.end();
  for (; decl_pos != decl_end; ++decl_pos)
  {
    SgFunctionDeclaration* const func_decl = decl_pos->first;

    SgSymbol* symb = func_decl->search_for_symbol_from_symbol_table();
    KaapiTaskAttribute* kta = (KaapiTaskAttribute*)
      symb->getAttribute("kaapitask");
    // assume(kta);

    if (kta->is_signature) continue ;

    if (DoKaapiModeAnalysis(project, kta) == false)
      is_success = false;
  }

  return is_success;
}
