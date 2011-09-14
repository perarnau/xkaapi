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
#include <sstream>
#include <string>
#include <map>
#include <set>
#include <vector>
#include <sys/types.h>
#include "rose_headers.h"
#include "utils.h"
#include "globals.h"
#include "kaapi_c2c_task.h"
#include "kaapi_abort.h"


bool KaapiTaskAttribute::hasReduction() const
{
  {
    std::vector<KaapiTaskFormalParam>::const_iterator pos = formal_param.begin();
    std::vector<KaapiTaskFormalParam>::const_iterator end = formal_param.end();
    for (; pos != end; ++pos) if (pos->mode == KAAPI_CW_MODE) return true;
  }
  {
    std::vector<KaapiTaskFormalParam>::const_iterator pos = extra_param.begin();
    std::vector<KaapiTaskFormalParam>::const_iterator end = extra_param.end();
    for (; pos != end; ++pos) if (pos->mode == KAAPI_CW_MODE) return true;
  }
  return false;
}


void KaapiTaskAttribute::buildReductionSet
(std::set<SgVariableSymbol*>& symbol_set)
{
  {
    std::vector<KaapiTaskFormalParam>::const_iterator pos = formal_param.begin();
    std::vector<KaapiTaskFormalParam>::const_iterator end = formal_param.end();
    for (; pos != end; ++pos)
    {
      if (pos->mode != KAAPI_CW_MODE) continue ;

      SgVariableSymbol* const sym =
	isSgVariableSymbol(pos->initname->get_symbol_from_symbol_table());
      if (sym == NULL) continue ;
      symbol_set.insert(sym);
    }
  }
  {
    std::vector<KaapiTaskFormalParam>::const_iterator pos = extra_param.begin();
    std::vector<KaapiTaskFormalParam>::const_iterator end = extra_param.end();
    for (; pos != end; ++pos)
    {
      if (pos->mode != KAAPI_CW_MODE) continue ;

      SgVariableSymbol* const sym =
	isSgVariableSymbol(pos->initname->get_symbol_from_symbol_table());
      if (sym == NULL) continue ;
      symbol_set.insert(sym);
    }
  }
}


void KaapiTaskAttribute::buildReductionSet
(std::set<KaapiTaskFormalParam*>& param_set)
{
  // fixme: redundant with the above function
  {
    std::vector<KaapiTaskFormalParam>::iterator pos = formal_param.begin();
    std::vector<KaapiTaskFormalParam>::iterator end = formal_param.end();
    for (; pos != end; ++pos)
    {
      if (pos->mode != KAAPI_CW_MODE) continue ;
      param_set.insert(&(*pos));
    }
  }
  {
    std::vector<KaapiTaskFormalParam>::iterator pos = extra_param.begin();
    std::vector<KaapiTaskFormalParam>::iterator end = extra_param.end();
    for (; pos != end; ++pos)
    {
      if (pos->mode != KAAPI_CW_MODE) continue ;
      param_set.insert(&(*pos));
    }
  }
}


SgClassDeclaration* KaapiTaskAttribute::buildInsertClassDeclaration
(
 SgGlobal* global_scope,
 SgScopeStatement* local_scope,
 SgClassDefinition* class_def
)
{
  // if it does not exist yet, build a class
  // declaration containing the formal params

  if (class_decl != 0) return class_decl;

  // build the variable symbol set from param vectors
  std::set<SgVariableSymbol*> symbol_set;
  buildReductionSet(symbol_set);

  // build type from symbol set
  class_decl = buildOutlineArgStruct
    (symbol_set, global_scope, class_def, "__kaapi_task_result_", false);
  if (class_decl == NULL) return NULL;

  // insert the declaration statement
  SgFunctionDeclaration* const func_decl = 
    SageInterface::getEnclosingFunctionDeclaration(local_scope, true);
  if (func_decl == NULL) KaapiAbort("func_decl == NULL");
  SageInterface::insertStatement(func_decl, class_decl);

  return class_decl;
}


SgFunctionDeclaration* KaapiTaskAttribute::buildInsertReducer
(SgType* work_type, SgType* result_type, SgGlobal* global_scope)
{
  // build a reducer function. this function iterate over the
  // variable symbol set and call the corresponding reducer

  static unsigned int id = 0;

  if (reducer_decl) return reducer_decl;

  SgFunctionParameterList* const param_list =
    SageBuilder::buildFunctionParameterList();

  SageInterface::appendArg
    (
     param_list, 
     SageBuilder::buildInitializedName
     (
      "__kaapi_sc",
      SageBuilder::buildPointerType(kaapi_stealcontext_ROSE_type)
      )
     );

  SageInterface::appendArg
    (
     param_list, 
     SageBuilder::buildInitializedName
     (
      "__kaapi_targ",
      SageBuilder::buildPointerType(SageBuilder::buildVoidType())
      )
     );

  SageInterface::appendArg
    (
     param_list, 
     SageBuilder::buildInitializedName
     (
      "__kaapi_tdata",
      SageBuilder::buildPointerType(SageBuilder::buildVoidType())
      )
     );

  SageInterface::appendArg
    (
     param_list, 
     SageBuilder::buildInitializedName
     (
      "__kaapi_tsize",
      SageBuilder::buildOpaqueType("size_t", global_scope)
      )
     );

  SageInterface::appendArg
    (
     param_list, 
     SageBuilder::buildInitializedName
     (
      "__kaapi_varg",
      SageBuilder::buildPointerType(SageBuilder::buildVoidType())
      )
     );

  std::ostringstream reducer_name;
  reducer_name << "__kaapi_reducer_" << (id++);
  reducer_decl = SageBuilder::buildDefiningFunctionDeclaration
    (
     reducer_name.str(),
     SageBuilder::buildIntType(),
     param_list,
     global_scope
     );
  reducer_decl->get_declarationModifier().get_storageModifier().setStatic();
  reducer_decl->set_endOfConstruct(SOURCE_POSITION);
  SageInterface::appendStatement
    (reducer_decl, isSgScopeStatement(global_scope));

  // build the reducer body.
  SgBasicBlock* const reducer_body = reducer_decl->get_definition()->get_body();

  // result_type* __kaapi_tw = (result_type*)__kaapi_tdata;
  SgVariableDeclaration* tw_decl = SageBuilder::buildVariableDeclaration
    ( 
     "__kaapi_tw", 
     SageBuilder::buildPointerType(result_type),
     SageBuilder::buildAssignInitializer
     (
      SageBuilder::buildCastExp
      (
       SageBuilder::buildVarRefExp("__kaapi_tdata", reducer_body),
       SageBuilder::buildPointerType(result_type)
       ),
      0
      ),
     reducer_body
      );
  SageInterface::appendStatement(tw_decl, reducer_body);

  // work_type* __kaapi_vw = (work_type*)__kaapi_varg;
  SgVariableDeclaration* vw_decl = SageBuilder::buildVariableDeclaration
    ( 
     "__kaapi_vw", 
     SageBuilder::buildPointerType(work_type),
     SageBuilder::buildAssignInitializer
     (
      SageBuilder::buildCastExp
      (
       SageBuilder::buildVarRefExp("__kaapi_varg", reducer_body),
       SageBuilder::buildPointerType(work_type)
       ),
      0
      ),
     reducer_body
      );
  SageInterface::appendStatement(vw_decl, reducer_body);

  // for each reduction variable, apply redop(tw->p_xxx, &vw->xxx)
  std::set<KaapiTaskFormalParam*> param_set;
  buildReductionSet(param_set);
  std::set<KaapiTaskFormalParam*>::const_iterator pos = param_set.begin();
  std::set<KaapiTaskFormalParam*>::const_iterator end = param_set.end();
  for (; pos != end; ++pos)
  {
    // FIXME: for now, only dimensionless types, non
    // builtin operators supported by this function

    KaapiReduceOperator_t* const red_op = (*pos)->redop;
    if (red_op->isbuiltin) KaapiAbort("isbuiltin not yet supported");

    std::string lhs_name;
    lhs_name.append("__kaapi_vw->p_");
    lhs_name.append((*pos)->initname->get_name().str());

    std::string rhs_name;
    rhs_name.append("&__kaapi_tw->");
    rhs_name.append((*pos)->initname->get_name().str());

    SgExprStatement* const call_stmt = SageBuilder::buildFunctionCallStmt
      (
       red_op->name_reducor,
       SageBuilder::buildVoidType(), 
       SageBuilder::buildExprListExp
       (
	SageBuilder::buildVarRefExp(lhs_name, reducer_body),
	SageBuilder::buildVarRefExp(rhs_name, reducer_body)
	),
       reducer_body
       );

    SageInterface::appendStatement(call_stmt, reducer_body);
  }

  // return 0
  SgReturnStmt* const return_stmt = SageBuilder::buildReturnStmt
    (SageBuilder::buildIntVal(0));
  SageInterface::appendStatement(return_stmt, reducer_body);

  return reducer_decl;
}


// task format generator

static inline void KaapiGenerateMode
(std::ostream& fout, enum KaapiAccessMode_t mode)
{
  switch (mode) {
    case KAAPI_V_MODE: fout << "    KAAPI_ACCESS_MODE_V, "; break;
    case KAAPI_W_MODE: fout << "    KAAPI_ACCESS_MODE_W, "; break;
    case KAAPI_R_MODE: fout << "    KAAPI_ACCESS_MODE_R, "; break;
    case KAAPI_RW_MODE:fout << "    KAAPI_ACCESS_MODE_RW, "; break;
    case KAAPI_CW_MODE:fout << "    KAAPI_ACCESS_MODE_CW, "; break;
    default:
   break;
  }
}

/***/
static void RecGenerateGetDimensionExpression
(
 std::ostringstream& sout,
 KaapiTaskAttribute* kta,
 SgExpression* expr
)
{
  if (isSgVarRefExp(expr))
  {
    SgVarRefExp* var = isSgVarRefExp(expr);
    SgVariableSymbol * sym = var->get_symbol();
    /* look up sym->get_name() in the list of formal parameter of the declaration */
    std::map<std::string,int>::iterator iparam = kta->lookup.find( sym->get_name().str() );
    if (iparam == kta->lookup.end())
    {
      // dump since this may be a macro or something
      // we dont have access to the defintion of
      sout << sym->get_name().str();
    }
    else
    {
      sout << "((" 
	   << kta->formal_param[iparam->second].type->unparseToString() 
	   << ")" << "arg->f" << iparam->second;
      if (kta->formal_param[iparam->second].mode != KAAPI_V_MODE)
	sout << ".data";
      sout << ")";
    }
  }
  else if (isSgUnsignedLongVal(expr))
  {
    SgUnsignedLongVal* value = isSgUnsignedLongVal(expr);
    sout << value->get_value();
  }
  else if (isSgSizeOfOp(expr))
  {
    SgSizeOfOp* esizeof = isSgSizeOfOp(expr);
    sout << " sizeof( ";
    if (esizeof->get_operand_expr() !=0)
      RecGenerateGetDimensionExpression(sout, kta, esizeof->get_operand_expr() );
    else /* here ??? */
      sout << SageInterface::get_name(esizeof->get_type());
  }
  else if (isSgCastExp(expr))
  {
    SgCastExp* cast = isSgCastExp(expr);
    sout << " ( " << SageInterface::get_name(cast->get_type()) << ")";
    RecGenerateGetDimensionExpression(sout, kta, cast->get_originalExpressionTree());
  }
  else if (isSgMultiplyOp(expr))
  {
    SgMultiplyOp* op = isSgMultiplyOp(expr);
    sout << "(";
    RecGenerateGetDimensionExpression(sout, kta, op->get_lhs_operand());
    sout << " * ";
    RecGenerateGetDimensionExpression(sout, kta, op->get_rhs_operand());
    sout << ")";
  }
  else if (isSgDivideOp(expr))
  {
    SgDivideOp* op = isSgDivideOp(expr);
    sout << "(";
    RecGenerateGetDimensionExpression(sout, kta, op->get_lhs_operand());
    sout << " / ";
    RecGenerateGetDimensionExpression(sout, kta, op->get_rhs_operand());
    sout << ")";
  }
  else if (isSgModOp(expr))
  {
    SgModOp* op = isSgModOp(expr);
    sout << "(";
    RecGenerateGetDimensionExpression(sout, kta, op->get_lhs_operand());
    sout << " % ";
    RecGenerateGetDimensionExpression(sout, kta, op->get_rhs_operand());
    sout << ")";
  }
  else if (isSgAddOp(expr))
  {
    SgAddOp* op = isSgAddOp(expr);
    sout << "(";
    RecGenerateGetDimensionExpression(sout, kta, op->get_lhs_operand());
    sout << " + ";
    RecGenerateGetDimensionExpression(sout, kta, op->get_rhs_operand());
    sout << ")";
  }
  else if (isSgSubtractOp(expr))
  {
    SgSubtractOp* op = isSgSubtractOp(expr);
    sout << "(";
    RecGenerateGetDimensionExpression(sout, kta, op->get_lhs_operand());
    sout << " - ";
    RecGenerateGetDimensionExpression(sout, kta, op->get_rhs_operand());
    sout << ")";
  }
  else {
    std::cerr << "Found an expression: @expr:" << expr << "  " << (expr == 0? "" : expr->class_name()) << std::endl;
    KaapiAbort("*** Bad expression for dimension");
  }
}

/***/
std::string GenerateGetDimensionExpression
(
 KaapiTaskAttribute* kta, 
 SgExpression* expr
)
{
  std::ostringstream expr_str;
  RecGenerateGetDimensionExpression( expr_str, kta, expr );
  return expr_str.str();
}

/***/
static std::string GenerateSetDimensionExpression
(
 KaapiTaskAttribute* kta, 
 SgExpression* expr,
 int index_inview
)
{
  if (index_inview != -1) return "";
  /* only generate change in the lda field */
  std::ostringstream expr_str;

  if (isSgVarRefExp(expr))
  {
    SgVarRefExp* var = isSgVarRefExp(expr);
    SgVariableSymbol * sym = var->get_symbol();
    /* look up sym->get_name() in the list of formal parameter of the declaration */
    std::map<std::string,int>::iterator iparam = kta->lookup.find( sym->get_name().str() );
    if (iparam == kta->lookup.end())
      KaapiAbort("****[kaapi_c2c] Cannot find which parameter is involved in lda expression");
    expr_str << "arg->f" << iparam->second;
    expr_str << " = view->lda;\n";
  }
  else {
    std::cerr << "****[kaapi_c2c] error. Cannot found lhs expression type for lda: @expr:" 
              << expr << "  " << (expr == 0? "" : expr->class_name()) << std::endl;
    KaapiAbort("****[kaapi_c2c] Expression for lda should be an identifier.");
  }

  return expr_str.str();
}

void DoKaapiGenerateFormat( std::ostream& fout, KaapiTaskAttribute* kta)
{
#if CONFIG_ENABLE_DEBUG
  std::cout << "****[kaapi_c2c] Task's generation format for function: " << kta->func_decl->get_name().str()
            << std::endl;
#endif // CONFIG_ENABLE_DEBUG
  
  kta->name_format = kta->name_paramclass + "_format";
  size_t cnt_param   = kta->formal_param.size() + (kta->has_retval ? +1 : 0) + (kta->has_this ? +1 : 0);
  size_t case_retval;
  size_t case_this;
  case_retval = (kta->has_retval ? kta->formal_param.size() : -1);
  case_this   = (kta->has_this ? 1+case_retval : -1);


  fout << "/*** Format for task argument:" 
       << kta->name_paramclass
       << "***/\n";
  
  SgUnparse_Info* sg_info = new SgUnparse_Info;
  sg_info->unset_forceQualifiedNames();

  std::string name_paramclass;
  if (SageInterface::is_C_language())
    name_paramclass = kta->name_paramclass;
  else if (SageInterface::is_Cxx_language())
    name_paramclass = kta->paramclass->get_qualified_name();
  
/* Not used: because generated into the translation unit
  fout << kta->paramclass->unparseToString(sg_info) << std::endl;
  fout << kta->typedefparamclass->unparseToString(sg_info) << std::endl << std::endl;
  fout << kta->fwd_wrapper_decl->unparseToString(sg_info) << std::endl << std::endl;
*/
  
  /* format definition */
  fout << "/* format object*/\n"
       << "struct kaapi_format_t*" << kta->name_format << " = 0;\n\n\n";

  /* format::get_count_params */
  fout << "size_t " << kta->name_format << "_get_count_params(const struct kaapi_format_t* fmt, const void* sp)\n"
       << "{ return " << cnt_param << "; }\n"
       << std::endl;

  /* format::get_mode_param */
  fout << "kaapi_access_mode_t " << kta->name_format << "_get_mode_param(const struct kaapi_format_t* fmt, unsigned int i, const void* sp)\n"
       << "{ \n"
       << "  static kaapi_access_mode_t mode_param[] = {\n";
  for (unsigned int i=0; i < kta->formal_param.size(); ++i)
    KaapiGenerateMode(fout, kta->formal_param[i].mode);

#if 1 // handle return value
  if (kta->has_retval) KaapiGenerateMode(fout, kta->retval.mode);
#endif // handle return value

#if 1 // handle this value
  if (kta->has_this) KaapiGenerateMode(fout, kta->thisval.mode);
#endif // handle return value

  fout << "    KAAPI_ACCESS_MODE_VOID\n  };\n"; /* marker for end of mode */
  fout << "  return mode_param[i];\n"
       << "}\n" 
       << std::endl;
  
  /* format::get_off_param */
  fout << "void* " << kta->name_format << "_get_off_param(const struct kaapi_format_t* fmt, unsigned int i, const void* sp)\n"
       << "{\n  " << name_paramclass << "* arg = (" << name_paramclass << "*)sp;\n"
       << "  switch (i) {\n";
  for (unsigned int i=0; i < kta->formal_param.size(); ++i)
  {
    fout << "    case " << i << ": return &arg->f" << i << ";\n";
  }
#if 1 // handle return value
  if (kta->has_retval)
    fout << "    case " << case_retval << ": return &arg->r;\n";
#endif // handle return value
#if 1 // handle this value
  if (kta->has_this)
    fout << "    case " << case_this << ": return &arg->thisfield;\n";
#endif // handle this value
  fout << "  }\n"
       << "  return 0;\n"
       << "}\n"
       << std::endl;

  /* format::get_access_param */
  fout << "kaapi_access_t " << kta->name_format << "_get_access_param(const struct kaapi_format_t* fmt, unsigned int i, const void* sp)\n"
       << "{\n  " << name_paramclass << "* arg = (" << name_paramclass << "*)sp;\n"
       << "  kaapi_access_t retval = {0,0};\n"
       << "  switch (i) {\n";
  for (unsigned int i=0; i < kta->formal_param.size(); ++i)
  {
    if (kta->formal_param[i].mode == KAAPI_V_MODE)
      fout << "    case " << i << ": break;\n";
    else
      fout << "    case " << i << ": retval = arg->f" << i << "; break; \n" ;/* because it is an access here */
  }

#if 1 // handle return value
  if (kta->has_retval)
    fout << "    case " << case_retval << ": retval = arg->r;  break;\n";
#endif // handle return value
#if 1 // handle this value
  if (kta->has_this)
    fout << "    case " << case_this << ": retval = arg->thisfield;  break;\n";
#endif // handle this value

  fout << "  }\n"
       << "  return retval;\n"
       << "}\n"
       << std::endl;
  
  /* format::set_access_param */
  fout << "void " << kta->name_format << "_set_access_param(const struct kaapi_format_t* fmt, unsigned int i, void* sp, const kaapi_access_t* a)\n"
       << "{\n  " << name_paramclass << "* arg = (" << name_paramclass << "*)sp;\n"
       << "  kaapi_access_t retval = {0,0};\n"
       << "  switch (i) {\n";
  for (unsigned int i=0; i < kta->formal_param.size(); ++i)
  {
    if (kta->formal_param[i].mode != KAAPI_V_MODE)
      fout << "    case " << i << ": arg->f" << i << " = *a" << "; return; \n"; /* because it is an access here */
  }

#if 1 // handle return value
  if (kta->has_retval)
    // we know this is an access
    fout << "    case " << case_retval << ": arg->r = *a; return; \n";
#endif // handle return value
#if 1 // handle this value
  if (kta->has_this)
    // we know this is an access
    fout << "    case " << case_this << ": arg->thisfield = *a; return; \n";
#endif // handle this value

  fout << "  }\n"
       << "}\n"
       << std::endl;

  /* format::get_fmt_param */
  fout << "const struct kaapi_format_t* " << kta->name_format << "_get_fmt_param(const struct kaapi_format_t* fmt, unsigned int i, const void* sp)\n"
       << "{\n  " << name_paramclass << "* arg = (" << name_paramclass << "*)sp;\n"
       << "  switch (i) {\n";
  for (unsigned int i=0; i < kta->formal_param.size(); ++i)
    fout << "    case " << i << ": return " << kta->formal_param[i].kaapi_format << ";\n";

#if 1 // handle return value
  if (kta->has_retval)
    fout << "    case " << case_retval << ": return " << kta->retval.kaapi_format << ";\n";
#endif // handle return value
#if 1 // handle this value
  if (kta->has_this)
    fout << "    case " << case_this << ": return " << kta->thisval.kaapi_format << ";\n";
#endif // handle this value

  fout << "  }\n"
       << "}\n"
       << std::endl;
       
  /* format::get_view_param */
  fout << "kaapi_memory_view_t " << kta->name_format << "_get_view_param(const struct kaapi_format_t* fmt, unsigned int i, const void* sp)\n"
       << "{\n  " << name_paramclass << "* arg = (" << name_paramclass << "*)sp;\n"
       << "  switch (i) {\n";
  for (unsigned int i=0; i < kta->formal_param.size(); ++i)
  {
    if (kta->formal_param[i].mode == KAAPI_V_MODE)
    {
      if (kta->israngedecl[i] <= 1)
      {
        SgUnparse_Info sgi;
        sgi.set_SkipDefinition();
        fout << "    case " << i << ": return kaapi_memory_view_make1d( 1, sizeof("
             << kta->formal_param[i].type->unparseToString(&sgi)
             << "));\n";
      }
      else { /* end of range type == int */
        fout << "    case " << i << ": return kaapi_memory_view_make1d( 1, sizeof(int));\n"; 
      }
    }
    else 
    {
      SgPointerType* ptrtype = isSgPointerType(kta->formal_param[i].type);
      if (ptrtype ==0) KaapiAbort("**** Error: bad internal assertion");
      SgType* type = ptrtype->get_base_type();
      // TODO here: add definition of the type, else we cannot compile it */
      while (isSgTypedefType( type ))
      {
        type = isSgTypedefType(type)->get_base_type();
      }
      
      if (kta->formal_param[i].attr->type == KAAPI_ARRAY_NDIM_TYPE)
      {          
        if (kta->formal_param[i].attr->dim == 0) /* in fact single element */
        {
          fout << "    case " << i << ": return kaapi_memory_view_make1d( 1" 
                              << ", sizeof(" << type->unparseToString() << "));\n";
        } 
        else if (kta->formal_param[i].attr->dim == 1)
        {
          fout << "    case " << i << ": return kaapi_memory_view_make1d( " 
                              << GenerateGetDimensionExpression(kta, kta->formal_param[i].attr->ndim[0])
                              << ", sizeof(" << type->unparseToString() << "));\n";
        } 
        else if (kta->formal_param[i].attr->dim == 2)
        {
          fout << "    case " << i << ": return kaapi_memory_view_make2d( "
                              << GenerateGetDimensionExpression(kta, kta->formal_param[i].attr->ndim[0]) << ","
                              << GenerateGetDimensionExpression(kta, kta->formal_param[i].attr->ndim[1]) << ",";
          if (kta->formal_param[i].attr->lda == 0) /* means contigous & row major == lda = dimj */
            fout              << GenerateGetDimensionExpression(kta, kta->formal_param[i].attr->ndim[1]);
          else
            fout              << GenerateGetDimensionExpression(kta, kta->formal_param[i].attr->lda);
          fout                << ", sizeof(" << type->unparseToString() << "));\n";
        }
        else if (kta->formal_param[i].attr->dim == 3)
        {
          fout << "    case " << i << ": kaapi_abort();\n";
        }
      }
      else { /* this is a begin of a range */
        fout << "    case " << i << ": return kaapi_memory_view_make1d( " 
                            << "arg->f" << kta->formal_param[i].attr->index_secondbound
                            << ",  sizeof(" << type->unparseToString() << "));\n";
      }
    }
  }

#if 1 // handle return value
  if (kta->has_retval)
  {
    SgPointerType* const ptrtype = isSgPointerType(kta->retval.type);
    if (ptrtype ==0) KaapiAbort("**** Error: bad internal assertion");

    SgType* type = ptrtype->get_base_type();

/* test non necessaire: structure retval tjrs correct si has_retval ? */
    if (kta->retval.attr->type == KAAPI_ARRAY_NDIM_TYPE)
    {
      if (kta->retval.attr->dim == 0) /* in fact single element */
      {
        fout << "    case " << case_retval << ": return kaapi_memory_view_make1d( 1" 
             << ", sizeof(" << type->unparseToString() << "));\n";
      }
      else
      {
        KaapiAbort("**** Error: bad internal assertion");	
      }
    }
    else
    {
      KaapiAbort("**** Error: bad internal assertion");
    }
  }
#endif // handle return value
#if 1 // handle this value
  if (kta->has_this)
  {
	fout << "    case " << case_this << ": return kaapi_memory_view_make1d( 1" 
	     << ", sizeof(" << isSgNamedType(kta->thisval.type)->get_qualified_name().str() << "));\n";
  }
#endif // handle this value

  fout << "  }\n"
       << "}\n"
       << std::endl;


  /* format::get_view_param */
  fout << "void " << kta->name_format << "_set_view_param(const struct kaapi_format_t* fmt, unsigned int i, void* sp, const kaapi_memory_view_t* view)\n"
       << "{\n  " << name_paramclass << "* arg = (" << name_paramclass << "*)sp;\n"
       << "  switch (i) {\n";
  for (unsigned int i=0; i < kta->formal_param.size(); ++i)
  {
    if (kta->formal_param[i].mode != KAAPI_V_MODE)
    {
      if ((kta->formal_param[i].attr->dim == 2) && (kta->formal_param[i].attr->lda !=0))
        fout << "    case " << i << ": {\n "
             << "      " << GenerateSetDimensionExpression(kta, kta->formal_param[i].attr->lda, -1) << "\n"
             << "    return; }\n";
      else if (kta->formal_param[i].attr->dim == 3)
      {
        fout << "    case " << i << ": kaapi_abort();\n";
      }      
    }
  }
  fout << "  }\n"
       << "}\n"
       << std::endl;

  /* format::reducor */
  fout << "void " << kta->name_format << "_reducor(const struct kaapi_format_t* fmt, unsigned int i, void* sp, const void* v)\n"
       << "{\n  " << name_paramclass << "* arg = (" << name_paramclass << "*)sp;\n"
       << "  switch (i) {\n";
  for (unsigned int i=0; i < kta->formal_param.size(); ++i)
  {
    if (kta->formal_param[i].mode == KAAPI_CW_MODE)
    {
      fout << "    case " << i << ": {\n";
      SgPointerType* ptrtype = isSgPointerType(kta->formal_param[i].type);
      if (ptrtype ==0) KaapiAbort("**** Error: bad internal assertion");
      SgType* type = ptrtype->get_base_type();
      // TODO here: add definition of the type, else we cannot compile it */
      while (isSgTypedefType( type ))
      {
        type = isSgTypedefType(type)->get_base_type();
      }

      /* pointer to the data: var */
      fout << "      " << type->unparseToString() 
           << "* var = (" << type->unparseToString() << "*) arg->f" << i << ".data;\n"
           << "      const " << type->unparseToString() 
           << "* value = ( const " << type->unparseToString() << "*)v;\n"
           << "      kaapi_memory_view_t view = " << kta->name_format << "_get_view_param(fmt, i, sp);\n";
           
      KaapiReduceOperator_t* redop = kta->formal_param[i].redop;
      /* the name of the variable is known: 
         from the type, find the righ operator or function.
         If it is a builtin operator, we generate the code for the reduction
      */
      if (redop->isbuiltin)
      {
        if (kta->formal_param[i].attr->type == KAAPI_ARRAY_NDIM_TYPE)
        {
          if (kta->formal_param[i].attr->dim ==0)
          {
            fout << "      *var " << redop->name_reducor << " *value;\n";
          }
          else {
            /* nested loop */
            for (unsigned int k=0; k<kta->formal_param[i].attr->dim; ++k)
            {
              std::ostringstream varindexloop;
              varindexloop << "ikak" << k;
              fout << "      for ( unsigned int " << varindexloop.str() << "=0; " 
                   << varindexloop.str() << " < view.size[" << k << "]; "
                   << "++" << varindexloop.str() << " )\n"
                   << "      {\n";
            }

            /* body */
            fout << "          var[ikak" << kta->formal_param[i].attr->dim-1
                 << "] " << redop->name_reducor 
                 << " value[ikak" << kta->formal_param[i].attr->dim-1 << "];\n";

            /* nested loop */
            for (unsigned int k=0; k<kta->formal_param[i].attr->dim; ++k)
            {
              std::ostringstream varindexloop;
              varindexloop << "ikak" << k;
              fout << "      } /*" << varindexloop.str() << "*/\n";
              if ((k == 0) && (kta->formal_param[i].attr->dim ==2))
              {
                fout << "      var += view.kda;\n";
              }
            }
          } /* else dim >0 */
        } /* else it is range 1d */
        else {
          fout << "      for (unsigned int k=0; k < view.size[0]; ++k)\n"
               << "        *var++ " << redop->name_reducor << " *value++;\n";
        }
      }      
      else /* not a builtin: generate a call to the user defined function */
      {
          fout << "      for (unsigned int k=0; k < view.size[0]; ++k)\n"
               << "        " << redop->name_reducor << "( var++, value++);\n";        
      }
      fout << "    } break;\n";
    }
  }
  fout << "  }\n"
       << "}\n"
       << std::endl;

  /* format::redinit */
  //isIntegerType
  fout << "void " << kta->name_format << "_redinit(const struct kaapi_format_t* fmt, unsigned int i, const void* sp, void* v)\n"
       << "{\n  " << name_paramclass << "* arg = (" << name_paramclass << "*)sp;\n"
       << "  switch (i) {\n";
  for (unsigned int i=0; i < kta->formal_param.size(); ++i)
  {
    if (kta->formal_param[i].mode == KAAPI_CW_MODE)
    {
      fout << "    case " << i << ": {\n";
      SgPointerType* ptrtype = isSgPointerType(kta->formal_param[i].type);
      if (ptrtype ==0) KaapiAbort("**** Error: bad internal assertion");
      SgType* type = ptrtype->get_base_type();
#if 0 // NO MORE REQUIRED: generated code are included into the translation unit
      // TODO here: add definition of the type, else we cannot compile it */
      while (isSgTypedefType( type ))
      {
        type = isSgTypedefType(type)->get_base_type();
      }
#endif

      /* pointer to the data: var */
      fout << "      " << type->unparseToString() 
           << "* var = (" << type->unparseToString() << "*) v;\n"
           << "      kaapi_memory_view_t view = " << kta->name_format << "_get_view_param(fmt, i, sp);\n";
           
      KaapiReduceOperator_t* redop = kta->formal_param[i].redop;

      /* the name of the variable is known: 
         from the type, find the righ operator or function.
         If it is a builtin operator, we generate the code for the reduction
      */
      if (redop->isbuiltin)
      {
        if (kta->formal_param[i].attr->type == KAAPI_ARRAY_NDIM_TYPE)
        {
          if (kta->formal_param[i].attr->dim ==0)
          {
            fout << "      *var = " << redop->name_redinit << ";\n";
          }
          else {
            /* nested loop */
            for (unsigned int k=0; k<kta->formal_param[i].attr->dim; ++k)
            {
              std::ostringstream varindexloop;
              varindexloop << "ikak" << k;
              fout << "      for ( unsigned int " << varindexloop.str() << "=0; " 
                   << varindexloop.str() << " < view.size[" << k << "]; "
                   << "++" << varindexloop.str() << " )\n"
                   << "      {\n";
            }

            /* body */
            fout << "          var[ikak" << kta->formal_param[i].attr->dim-1
                 << "] = " << redop->name_redinit << ";\n";

            /* nested loop */
            for (unsigned int k=0; k<kta->formal_param[i].attr->dim; ++k)
            {
              std::ostringstream varindexloop;
              varindexloop << "ikak" << k;
              fout << "      } /*" << varindexloop.str() << "*/\n";
              if ((k == 0) && (kta->formal_param[i].attr->dim ==2))
              {
                fout << "      var += view.kda;\n";
              }
            }
          } /* else dim >0 */
        } /* else it is range 1d */
        else {
          fout << "      for (unsigned int k=0; k < view.size[0]; ++k)\n"
               << "        *var++ = " << redop->name_redinit << ";\n";
        }
      }
      else
      {
	fout << "      for (unsigned int k=0; k < view.size[0]; ++k, ++var)\n";
	if (redop->name_redinit.empty())
          fout << "         memset(var, 0, sizeof(*var));\n";
        else
	  fout << "        " << redop->name_redinit << "(var);\n";
      }
      fout << "    } break;\n";
    }
  }
  fout << "  }\n"
       << "}\n"
       << std::endl;


  /* format::get_task_binding */
  fout << "void " << kta->name_format << "_get_task_binding(const struct kaapi_format_t* fmt, const kaapi_task_t* t, kaapi_task_binding_t* tb)\n"
       << "{ return; }\n"
       << std::endl;

  std::string wrapper_decl_name;
  if (SageInterface::is_C_language())
    wrapper_decl_name = kta->wrapper_decl->get_name();
  else if (SageInterface::is_Cxx_language())
    wrapper_decl_name = kta->wrapper_decl->get_qualified_name();
       
  /* Generate constructor function that register the format */
  fout << "/* constructor method */\n" 
       << "__attribute__ ((constructor)) void " << kta->name_format << "_constructor(void)\n"
       << "{\n"
       << "  if (" << kta->name_format << " !=0) return;\n"
       << "  " << kta->name_format << " = kaapi_format_allocate();\n"
       << "  kaapi_format_taskregister_func(\n"
       << "    " << kta->name_format << ",\n" /* format object */
       << "    " << wrapper_decl_name << ",\n" /* body */
       << "    " << 0 << ",\n" /* bodywh */
       << "    \"" << kta->name_format << "\",\n" /* name */
       << "    sizeof(" << name_paramclass << "),\n" /* sizeof the arg struct */
       << "    " << kta->name_format << "_get_count_params,\n" /* get_count_params */
       << "    " << kta->name_format << "_get_mode_param,\n" /* get_mode_param */
       << "    " << kta->name_format << "_get_off_param,\n" /* get_off_param */
       << "    " << kta->name_format << "_get_access_param,\n" /* get_access_param */
       << "    " << kta->name_format << "_set_access_param,\n" /* set_access_param */
       << "    " << kta->name_format << "_get_fmt_param,\n" /* get_fmt_param */
       << "    " << kta->name_format << "_get_view_param,\n" /* get_view_param */
       << "    " << kta->name_format << "_set_view_param,\n" /* set_view_param */
       << "    " << kta->name_format << "_reducor,\n" /* reducor */
       << "    " << kta->name_format << "_redinit,\n" /* reducor */
       << "    " << kta->name_format << "_get_task_binding\n" /* get_task_binding */
       << "  );\n"
       << "}\n"
       << std::endl;
}
