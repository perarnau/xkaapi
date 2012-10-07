/*
** xkaapi
** 
**
** Copyright 2009,2010,2011,2012 INRIA.
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
#include <stdexcept>
#include <ctype.h>
#include "parser.h"
#include "globals.h"
#include "utils.h"
#include "kaapi_pragma.h"
#include "kaapi_c2c_task.h"
#include "kaapi_abort.h"
#include "rose_headers.h"
#include "kaapi_loop.h"


static inline int isletter( char c)
{
  return (c == '_') || isalpha(c);
}

static inline int isletternum( char c)
{
  return (c == '_') || isalnum(c);
}


void Parser::skip_ws()
{
  while ((*rpos == ' ') || (*rpos == '\t') || (*rpos == '\n')) ++rpos;
}


char Parser::readchar() throw(std::overflow_error)
{
  if (rpos == rlast) 
  {
    ++rpos;
    return EOF;
  }
  if (rpos > rlast) throw std::overflow_error("empty buffer");
  return *rpos++;
}


void Parser::putback()
{
  --rpos;
}


/** #pragma kaapi task parsing
*/
void Parser::DoKaapiPragmaTask( SgPragmaDeclaration* sgp )
{
  Sg_File_Info* fileInfo = sgp->get_file_info();

  /* #pragma kaapi task: find next function declaration to have the name */
  if (sgp->get_parent() ==0)
    KaapiAbort( "*** Internal compiler error: mal formed AST" );
    
  /* it is assumed that the statement following the #pragma kaapi task is a declaration */
  SgStatement* fnode = SageInterface::getNextStatement ( sgp );

  std::string pragma_string;
  ParseGetLine( pragma_string );

  if (isSgTemplateDeclaration(fnode))
  {
    if (isSgTemplateDeclaration(fnode)->get_template_kind() == SgTemplateDeclaration::e_template_function)
      fnode->setAttribute("kaapi_templatetask", new KaapiPragmaString(pragma_string));
    return;
  }
  
  SgFunctionDeclaration* functionDeclaration = isSgFunctionDeclaration( fnode );
  if (functionDeclaration ==0) 
  {
    std::cerr << "In filename '" << fileInfo->get_filename() << "' LINE: " << fileInfo->get_line()
              << "\n#pragma kaapi task must be followed by a function declaration. Found: " << fnode->class_name()
              << std::endl;
    KaapiAbort("**** error");
  }

#if CONFIG_ENABLE_DEBUG
std::cerr << "****[kaapi_c2c] Found #pragma kaapi task. "
          << " Function declaration: @" << functionDeclaration
          << " Function declaration Symbol: @" << functionDeclaration->search_for_symbol_from_symbol_table()  
          << std::endl;
#endif // CONFIG_ENABLE_DEBUG

 all_task_func_decl.push_back( std::make_pair(functionDeclaration, pragma_string) );
}

void Parser::DoKaapiPragmaSignature( SgPragmaDeclaration* sgp )
{
  DoKaapiPragmaTask(sgp);
  SgStatement* const next_stmt = SageInterface::getNextStatement(sgp);
  SgFunctionDeclaration* const func_decl = isSgFunctionDeclaration(next_stmt);
  if (func_decl) all_signature_func_decl.insert(func_decl);
}

/* */

std::string ConvertCType2KaapiFormat(SgType* type)
{
  switch (type->variantT()) 
  {
    case V_SgTypeChar: return "kaapi_char_format";
    case V_SgTypeSignedChar: return "kaapi_char_format";
    case V_SgTypeUnsignedChar: return "kaapi_uchar_format";

    case V_SgTypeShort: return "kaapi_short_format";
    case V_SgTypeSignedShort: return "kaapi_short_format";
    case V_SgTypeUnsignedShort: return "kaapi_ushort_format";

    case V_SgTypeInt: return "kaapi_int_format";
    case V_SgTypeSignedInt: return "kaapi_int_format";
    case V_SgTypeUnsignedInt: return "kaapi_uint_format";

    case V_SgTypeLong: return "kaapi_long_format";
    case V_SgTypeSignedLong: return "kaapi_long_format";
    case V_SgTypeUnsignedLong: return "kaapi_ulong_format";

    case V_SgTypeLongLong: return "kaapi_longlong_format";
    case V_SgTypeSignedLongLong: return "kaapi_longlong_format";
    case V_SgTypeUnsignedLongLong: return "kaapi_ulonglong_format";

    case V_SgTypeFloat: return "kaapi_float_format";
    case V_SgTypeDouble: return "kaapi_double_format";
    case V_SgTypeLongDouble: return "kaapi_longdouble_format";

    case V_SgPointerType: return "kaapi_voidp_format";
    
    case V_SgTypedefType: return ConvertCType2KaapiFormat( isSgTypedefType(type)->get_base_type() );
    case V_SgModifierType: return ConvertCType2KaapiFormat( isSgModifierType(type)->get_base_type() );

    case V_SgEnumType: return "kaapi_int_format";

    default: {
      std::ostringstream sout;
      sout << "<NO KAAPI FORMAT for: " << type->class_name() << ">";
      return sout.str();
    }
  }
}

void Parser::DoKaapiPragmaLoop( SgPragmaDeclaration* sgp )
{
  Sg_File_Info* fileInfo = sgp->get_file_info();
#if CONFIG_ENABLE_DEBUG
  std::cerr << "****[kaapi_c2c] Found #pragma kaapi loop directive !!!!."
            << "     In filename '" << fileInfo->get_filename() 
            << "' LINE: " << fileInfo->get_line()
            << std::endl;
#endif // CONFIG_ENABLE_DEBUG
  SgForStatement* forloop = isSgForStatement(SageInterface::getNextStatement ( sgp ));
  if (forloop ==0)
  {
    std::cerr << "****[kaapi_c2c] #pragma kaapi loop directive must be followed by a for loop statement."
              << "     In filename '" << fileInfo->get_filename() 
              << "' LINE: " << fileInfo->get_line()
              << std::endl;
    KaapiAbort("**** error");
  }

  // parse as if it was a task declaration with
  // output and reduction clauses.
  // KaapiTaskDeclaration cannot be used since we
  // dont have a functionDeclaration for adaptive
  // loops (todo).

  KaapiTaskAttribute* kta = new KaapiTaskAttribute;
  kta->is_signature = false;
  kta->func_decl = NULL;
  kta->has_retval = false;
  kta->has_this = false;

  // __to_replace__
  std::string clause;
  ParseIdentifier(clause);

  if (clause == "reduction")
  {
    skip_ws();
    char c = readchar();
    if (c == EOF || c != '(') KaapiAbort("syntax error '('");

    while (true)
    {
      // parse reducer:variable pair
      skip_ws();
      std::string red_name;
      ParseIdentifier(red_name);

      if (red_name.empty())
      {
	/* try to read char -> basic operator +, - etc */
	char c;
	c = readchar();
	if ((c == '+') || (c == '-') || (c == '*') || (c == '^'))
	{
	  red_name = c;
	  c = readchar();
	}
	else if ((c == '&') || (c == '|'))
	{ /* may be && or || */
	  char c1;
	  c1 = readchar();
	  if (c != c1)
	  {
	    red_name = c;
	  }
	  else
	  {
	    if (c == '&') red_name = "&&";
	    else red_name = "||";
	  }
	}
	else
	{
	  KaapiAbort("invalid reduction operator");
	}
      }

      std::map<std::string,KaapiReduceOperator_t*>::iterator red_pos =
	kaapi_user_definedoperator.find(red_name);
      if (red_pos == kaapi_user_definedoperator.end())
	KaapiAbort("reduction operator not found");

      skip_ws();
      c = readchar();
      if (c != ':') KaapiAbort("syntax error ':'");

      skip_ws();
      std::string var_name;
      ParseIdentifier(var_name);

      SgVariableSymbol* const var_sym =
	SageInterface::lookupVariableSymbolInParentScopes(var_name, forloop);
      if (var_sym == NULL) KaapiAbort("variable not found\n");

      // build the corresponding parameter
      KaapiTaskFormalParam param;
      param.mode = KAAPI_CW_MODE;
      param.redop = red_pos->second;
      param.attr = new KaapiParamAttribute;
      param.attr->type = KAAPI_ARRAY_NDIM_TYPE;
      param.attr->storage = KAAPI_COL_MAJOR;
      param.attr->dim = 0;
      param.initname = var_sym->get_declaration();
      param.type = param.initname->get_type();
      param.kaapi_format = ConvertCType2KaapiFormat(param.type);

      // insert in the kta
      const unsigned int index = kta->formal_param.size();
      kta->lookup.insert(std::make_pair(var_name, index));
      kta->formal_param.push_back(param);
      kta->israngedecl.push_back(0);

      skip_ws();
      c = readchar();
      if (c != ',') break ;
    }
  }
  else if (clause.size()) KaapiAbort("clause not implemented");
  // __to_replace__

  SgFunctionDeclaration* entrypoint;
  SgFunctionDeclaration* splitter;
  SgClassDeclaration* contexttype;

  SgStatement* stmt = buildConvertLoop2Adaptative
    ( forloop, entrypoint, splitter, contexttype, kta );

  delete kta;

  if (stmt !=0)
  {
#if 0
    SageInterface::prependStatement(
      SageBuilder::buildNondefiningFunctionDeclaration
      ( entrypoint, forloop->get_scope() ),
      forloop->get_scope()
    );
#endif
    SageInterface::insertStatement( forloop, stmt );
    SageInterface::removeStatement( forloop );
  }
}


static inline bool is_signature(SgFunctionDeclaration* func_decl)
{
  if (all_signature_func_decl.find(func_decl) != all_signature_func_decl.end())
    return true;
  return false;
}

void Parser::DoKaapiTaskDeclaration
  ( SgFunctionDeclaration* functionDeclaration )
{
  Sg_File_Info* fileInfo = functionDeclaration->get_file_info();

  /* */
  KaapiTaskAttribute* kta = new KaapiTaskAttribute;
  kta->is_signature = is_signature(functionDeclaration);
  kta->func_decl = functionDeclaration;
  kta->has_retval = false;

  /* Mark the body of the function as a "kaapiisparallelregion" */
  if (functionDeclaration->get_definition() !=0)
  {
    SgBasicBlock* body = functionDeclaration->get_definition()->get_body();
    body->setAttribute("kaapiisparallelregion", (AstAttribute*)-1);
#if 0
    std::cout << "Set the body:" << body << " of the function: "<<functionDeclaration->get_name().str() 
              << " as an implicit parallel region"
              << std::endl;
#endif
  }

  /* Store formal parameter in the KaapiTaskAttribute structure */
  SgFunctionParameterList* func_param = functionDeclaration->get_parameterList();
  SgInitializedNamePtrList& args = func_param->get_args();

  const unsigned int args_size = args.size();
  SgScopeStatement* scope_declaration = functionDeclaration->get_scope();
  kta->formal_param.resize( args_size );
  kta->israngedecl.resize( args_size );
  for (unsigned int i=0; i<args.size(); ++i)
  {
    SgInitializedName* initname   = args[i];
    kta->formal_param[i].initname = initname; //->get_qualified_name().str();
    kta->formal_param[i].type     = initname->get_typeptr();
    if (initname->get_name().is_null())
    {
      std::cerr <<  "In filename '" << fileInfo->get_filename() << "' LINE: " << fileInfo->get_line()
                << " #pragma kaapi task: function '" << functionDeclaration->get_mangled_name().str() << "'"
                << " has missing name for its " << i << "-th formal parameter"
                << std::endl;
      KaapiAbort("**** error");
    }
    /* try to find the parameter through the scope */
    kta->lookup.insert( std::make_pair(initname->get_name().str(), i ) );
  }

  /* parse the end of the stream of declaration
  */
  ParseListParam( fileInfo, kta, scope_declaration );

  /* Add the class declaration for the parameters */
  kta->name_paramclass = std::string("__kaapi_args_") + 
    functionDeclaration->get_name().str() + 
    functionDeclaration->get_mangled_name().str();

  kta->paramclass = 
    buildClassDeclarationAndDefinition( 
      kta->name_paramclass,
      scope_declaration
    );

  SgVariableDeclaration* thismemberDeclaration = 0;
  
  /* Add member for each parameter */
  for (unsigned int i=0; i<kta->formal_param.size(); ++i)
  {
    std::ostringstream name;
    name << "f" << i;
    SgType* member_type = 0;
redo_selection:
    if (kta->formal_param[i].mode == KAAPI_V_MODE) 
    {
      member_type = kta->formal_param[i].type;
      if (isSgModifierType(member_type))
        member_type = isSgModifierType(member_type)->get_base_type();
      kta->formal_param[i].kaapi_format = ConvertCType2KaapiFormat(kta->formal_param[i].type);
    }
    else if (kta->formal_param[i].mode == KAAPI_VOID_MODE) 
    {
        std::cerr << "****[kaapi_c2c] Error: undefined access mode for parameter '" << kta->formal_param[i].initname->get_name().str() << "'\n"
                  << "     In filename '" << fileInfo->get_filename() << "' LINE: " << fileInfo->get_line()
                  << std::endl;
        KaapiAbort("**** error");    } 
    else 
    {
      if (!isSgPointerType(kta->formal_param[i].type))
      { /* read/write/reduction should be pointer: else move them to be by value */
#if 1 // TG: CONFIG_ENABLE_DEBUG
        std::cerr << "****[kaapi_c2c] Warning: incorrect access mode: not a pointer type. \n"
                  << "                         Change access mode declaration to value.\n"
                  << "     In filename '" << fileInfo->get_filename() << "' LINE: " << fileInfo->get_line()
                  << " formal parameter '" << kta->formal_param[i].initname->get_name().str()
                  << "' is declared as read/write/reduction but is not a pointer type. Move it as declared as a value.\n"
                  << std::endl;
#endif // CONFIG_ENABLE_DEBUG
        kta->formal_param[i].mode = KAAPI_V_MODE;
        goto redo_selection;
      }
      if (kta->israngedecl[i] <= 1) /* 2 was the size */
      {
        kta->formal_param[i].kaapi_format = ConvertCType2KaapiFormat(
          isSgPointerType(kta->formal_param[i].type)->get_base_type()
        );
        member_type = kaapi_access_ROSE_type;
      }
      else {
        /* This is the end bound of a 1D range: store an integer in place of a type */
        kta->formal_param[i].mode = KAAPI_V_MODE;
        kta->formal_param[i].kaapi_format = "kaapi_int_format";
        member_type = SageBuilder::buildIntType();
      }
    }

    SgVariableDeclaration* memberDeclaration =
      SageBuilder::buildVariableDeclaration (
        name.str(), 
        member_type,
        0, 
        kta->paramclass->get_definition()
    );
    memberDeclaration->set_endOfConstruct(SOURCE_POSITION);
    
    kta->paramclass->get_definition()->append_member(memberDeclaration);
  }
  
  SgMemberFunctionDeclaration* memberDecl = isSgMemberFunctionDeclaration(functionDeclaration);
  if (memberDecl !=0)
  {
    /* add this as extra parameter */
    std::ostringstream name;
    name << "thisfield";
    SgType* member_type = memberDecl->get_associatedClassDeclaration()->get_type();
    kta->has_this = true;

    thismemberDeclaration =
      SageBuilder::buildVariableDeclaration (
        name.str(), 
        kaapi_access_ROSE_type,
        0, 
        kta->paramclass->get_definition()
    );
    thismemberDeclaration->set_endOfConstruct(SOURCE_POSITION);
    
    kta->paramclass->get_definition()->append_member(thismemberDeclaration);

    KaapiTaskFormalParam& param = kta->thisval;
    
    SgMemberFunctionType* methodType = isSgMemberFunctionType( memberDecl->get_type() );
    if (methodType == 0) KaapiAbort("Internal error");

    if (memberDecl->get_functionModifier().isVirtual())
    {
      std::cerr << "****[kaapi_c2c] Error: Task cannot be defined on virtual method. \n"
                << "     In filename '" << fileInfo->get_filename() << "' LINE: " << fileInfo->get_line()
                << std::endl;
      KaapiAbort("**** error");
    }
    
    /* mode must be R if const method, else RW, else specified in pragma */
    if (methodType->isConstFunc())
      param.mode          = KAAPI_R_MODE;
    else
      param.mode          = KAAPI_RW_MODE;
    param.redop         = 0;
    param.attr          = new KaapiParamAttribute;
    param.attr->type    = KAAPI_ARRAY_NDIM_TYPE;
    param.attr->storage = KAAPI_COL_MAJOR;
    param.attr->dim     = 0;
    param.type          = member_type;          /* type of the class */
    param.kaapi_format  = "kaapi_voidp_format"; /* should be the class format */
  }

  /* Add extra parameter */
  for (unsigned int i=0; i<kta->extra_param.size(); ++i)
  {
    std::ostringstream name;
    name << "e" << i;
    SgType* member_type = 0;

    member_type = kta->extra_param[i].type;
    if (isSgModifierType(member_type)) 
    {
      member_type = isSgModifierType(member_type)->get_base_type();
      kta->formal_param[i].kaapi_format = ConvertCType2KaapiFormat(kta->formal_param[i].type);
    }

    SgVariableDeclaration* memberDeclaration =
      SageBuilder::buildVariableDeclaration (
        name.str(), 
        member_type,
        0, 
        kta->paramclass->get_definition()
    );
    memberDeclaration->set_endOfConstruct(SOURCE_POSITION);
    
    kta->paramclass->get_definition()->append_member(memberDeclaration);
  }

#if 1 // handle return value
  SgType* ret_type = functionDeclaration->get_orig_return_type();
  if (isSgTypeVoid(ret_type) == NULL)
  {
    static const char* const retval_name = "__kaapi_retval";

    SgPointerType* const ret_ptrtype =
      SageBuilder::buildPointerType(ret_type);

    SgInitializedName* const init_name =
      new SgInitializedName(SgName(retval_name), ret_ptrtype);

    kta->has_retval = true;

    KaapiTaskFormalParam& param = kta->retval;

    param.mode = KAAPI_W_MODE;
    param.redop = NULL;
    param.attr = new KaapiParamAttribute;
    param.attr->type = KAAPI_ARRAY_NDIM_TYPE;
    param.attr->storage = KAAPI_COL_MAJOR;
    param.attr->dim = 0;
    param.initname = init_name;
    param.type = ret_ptrtype;
    param.kaapi_format = ConvertCType2KaapiFormat
      (isSgPointerType(ret_ptrtype)->get_base_type());

    kta->lookup.insert
      (std::make_pair(init_name->get_name().str(), args_size));

    std::ostringstream member_name;
    member_name << "r";

    SgVariableDeclaration* const member_decl =
      SageBuilder::buildVariableDeclaration
      (
       member_name.str(), 
       kaapi_access_ROSE_type,
       0,
       kta->paramclass->get_definition()
       );
    member_decl->set_endOfConstruct(SOURCE_POSITION);
    kta->paramclass->get_definition()->append_member(member_decl);
  }
#endif // handle return value

  kta->typedefparamclass = SageBuilder::buildTypedefDeclaration(
      kta->name_paramclass, 
      kta->paramclass->get_type(), //->get_definingDeclaration()), //get_firstNondefiningDeclaration()), 
      scope_declaration
  );
  SageInterface::insertStatement( functionDeclaration, kta->paramclass, true );
  SageInterface::insertStatement( kta->paramclass, kta->typedefparamclass, false );
  
  /* generate the wrapper function and its fwd declaration. Copy the instructions
     to generate the function parameter list because it cannot be shared between
     2 delcarations and deep copy == seg fault
   */
  kta->mangled_name = std::string(functionDeclaration->get_name().str()) + 
      functionDeclaration->get_mangled_name().str();
  kta->name_wrapper = std::string("__kaapi_wrapper_") + kta->mangled_name;

  SgFunctionParameterList* fwd_parameterList = SageBuilder::buildFunctionParameterList();
  SageInterface::appendArg(fwd_parameterList, 
    SageBuilder::buildInitializedName("_k_arg", SageBuilder::buildPointerType(SageBuilder::buildVoidType()) )
  );
  SageInterface::appendArg(fwd_parameterList, 
    SageBuilder::buildInitializedName("_k_thread", SageBuilder::buildPointerType(kaapi_thread_ROSE_type) )
  );

  /* fwd declaration before the body of the original function 
     except for class method.
  */
  kta->fwd_wrapper_decl =
    SageBuilder::buildNondefiningFunctionDeclaration (
      kta->name_wrapper, 
      SageBuilder::buildVoidType(),
      fwd_parameterList,
      scope_declaration
  );
  if (!kta->has_this)
    ((kta->fwd_wrapper_decl->get_declarationModifier()).get_storageModifier()).setExtern();

  /* declaration + definition: generated after the #pragma kaapi task */
  SgFunctionParameterList* parameterList = SageBuilder::buildFunctionParameterList();

  SageInterface::appendArg(parameterList, 
    SageBuilder::buildInitializedName("_k_arg", SageBuilder::buildPointerType(SageBuilder::buildVoidType()) )
  );
  SageInterface::appendArg(parameterList, 
    SageBuilder::buildInitializedName("_k_thread", SageBuilder::buildPointerType(kaapi_thread_ROSE_type) )
  );
  kta->wrapper_decl =
    SageBuilder::buildDefiningFunctionDeclaration (
      kta->name_wrapper, 
      SageBuilder::buildVoidType(),
      parameterList, 
      scope_declaration
  );
  if (kta->has_this)
    ((kta->wrapper_decl->get_declarationModifier()).get_storageModifier()).setStatic();

//DO not put it extern, fwd declaration already indicates that
//  ((kta->wrapper_decl->get_declarationModifier()).get_storageModifier()).setExtern();
  SgBasicBlock*  wrapper_body = kta->wrapper_decl->get_definition()->get_body();
  SgClassType* class_forarg = new SgClassType(kta->paramclass->get_firstNondefiningDeclaration());
  
  /* add call to the original function */
  SgVariableDeclaration* truearg_decl = SageBuilder::buildVariableDeclaration ( 
    "thearg", 
    SageBuilder::buildPointerType(class_forarg),
    SageBuilder::buildAssignInitializer(
      SageBuilder::buildCastExp(
        SageBuilder::buildVarRefExp ("_k_arg", wrapper_body ),
        SageBuilder::buildPointerType(class_forarg)
      ),
      0
    ),
    wrapper_body
  );
  SageInterface::prependStatement( truearg_decl, wrapper_body );
  
  SgExprListExp* argscall = SageBuilder::buildExprListExp();
  
  for (unsigned int i=0; i<kta->formal_param.size(); ++i)
  {
    std::ostringstream fieldname;
    if (kta->israngedecl[i] <=1)
    {
      if (kta->formal_param[i].mode == KAAPI_V_MODE)
        fieldname << "thearg->f" << i;
      else {
        fieldname << "(" << kta->formal_param[i].type->unparseToString() << ")"
                   << "thearg->f" << i << ".data";
      }
    }
    else { /* bound end of a range store the size: rebuild the pointer */
      KaapiParamAttribute* kpa = kta->formal_param[i].attr;
      fieldname << "((" << kta->formal_param[kpa->index_firstbound].type->unparseToString() << ")"
                << "thearg->f" << kpa->index_firstbound << ".data)"
                << " + thearg->f" << i;
    }
    SageInterface::appendExpression(argscall,
      SageBuilder::buildOpaqueVarRefExp(fieldname.str(),wrapper_body)
    );
  }

  // generate: original_body(xxx) or thisfield->original_body(xxx)
  SgExprStatement* callStmt = 0;
  if (kta->has_this)
  {
    std::ostringstream fieldname;
    SgMemberFunctionDeclaration* memberDecl = isSgMemberFunctionDeclaration(functionDeclaration);
    fieldname << "((" << memberDecl->get_class_scope()->get_qualified_name().str() 
              << "*)(thearg->thisfield.data))->" 
              << memberDecl->get_name().str();
    callStmt = SageBuilder::buildFunctionCallStmt(
       fieldname.str(),
       SageBuilder::buildVoidType(), 
       argscall,
       wrapper_body
    );
  }
  else {
    callStmt = SageBuilder::buildFunctionCallStmt(
       functionDeclaration->get_name(),
       SageBuilder::buildVoidType(), 
       argscall,
       wrapper_body
    );
  }

  if (kta->has_retval)
  {
    SgAssignInitializer* const res_initializer =
      SageBuilder::buildAssignInitializer(callStmt->get_expression());

    static const char* const res_name = "res";
    SgVariableDeclaration* const res_decl = SageBuilder::buildVariableDeclaration
      (res_name, ret_type, res_initializer, wrapper_body);

    SageInterface::appendStatement(res_decl, wrapper_body);
    res_initializer->setAttribute("kaapiwrappercall", (AstAttribute*)-1);

    // generate: if (fu->r.data != NULL) *fu->r.data = res;
    // todo: iamlazymode == 1 ...

    std::ostringstream zero("0");
    SgExpression* const zero_expr = SageBuilder::buildOpaqueVarRefExp
      (zero.str(), wrapper_body);
    SgExpression* const null_expr = SageBuilder::buildCastExp
      (zero_expr, SageBuilder::buildPointerType(SageBuilder::buildVoidType()));

    SgPointerType* const ret_ptrtype =
      SageBuilder::buildPointerType(ret_type);

    SgVarRefExp* const field_expr = SageBuilder::buildOpaqueVarRefExp
      ("(void*)thearg->r.data", wrapper_body);
    SgNotEqualOp* const cond_stmt = SageBuilder::buildNotEqualOp
      (field_expr, null_expr);

    std::ostringstream fieldname;
    fieldname << "*((" << ret_ptrtype->unparseToString() << ")thearg->r.data)";
    SgVarRefExp* const retval_expr =
      SageBuilder::buildOpaqueVarRefExp(fieldname.str(), wrapper_body);
    SgExpression* const res_expr = SageBuilder::buildVarRefExp
      ("res", wrapper_body);
    SgExprStatement* const true_stmt = SageBuilder::buildAssignStatement
      (retval_expr, res_expr);

    SgStatement* const false_stmt = NULL;

    SgIfStmt* const if_stmt = SageBuilder::buildIfStmt
      (cond_stmt, true_stmt, false_stmt);
    SageInterface::appendStatement(if_stmt, wrapper_body);
  }
  else {
    callStmt->setAttribute("kaapiwrappercall", (AstAttribute*)-1);
    SageInterface::insertStatement( truearg_decl, callStmt, false );
  }
  
//  SageInterface::insertStatement( kta->typedefparamclass, kta->wrapper_decl, false );
  if (!kta->has_this)
    SageInterface::insertStatement( kta->typedefparamclass, kta->fwd_wrapper_decl, false );

  SageInterface::insertStatement( kta->func_decl, kta->wrapper_decl, false );

  /* annotated the AST function declaration attached symbol with the task attribute */
#if 0
  functionDeclaration->setAttribute("kaapitask", 
    kta
  );
#endif
  functionDeclaration->search_for_symbol_from_symbol_table()->setAttribute("kaapitask", 
    kta
  );
  all_tasks.push_back( kta );
  all_manglename2tasks.insert( std::make_pair(kta->mangled_name, kta) );
#if 0
  std::cout << "***** Attach Task attribut to declaration: " << functionDeclaration 
            << " name: " << functionDeclaration->get_name().str()
            << " access mode is:" << input.str()
            << std::endl;
#endif
}


/**
*/
void Parser::DoKaapiPragmaNoTask( SgPragmaDeclaration* sgp )
{
  /* #pragma kaapi task: find next function declaration to have the name */
  if (sgp->get_parent() ==0)
    KaapiAbort( "*** Internal compiler error: mal formed AST" );
    
  /* it is assumed that the statement following the #pragma kaapi task is a declaration */
  SgNode* fnode = SageInterface::getNextStatement( sgp );
#if 0
          sgp->get_parent()-> get_traversalSuccessorByIndex( 
              1+ sgp->get_parent()->get_childIndex( sgp ) );
#endif

  SgExprStatement* exprstatement = isSgExprStatement(fnode);
  if ((exprstatement ==0) || (isSgFunctionCallExp( exprstatement->get_expression() ) ==0))
  {
    Sg_File_Info* fileInfo = sgp->get_file_info();
    std::cerr <<  "In filename '" << fileInfo->get_filename() << "' LINE: " << fileInfo->get_line()
              << " #pragma kaapi notask must be followed by a function call expression"
              << " it is a:" << fnode->class_name()
              << std::endl;
    KaapiAbort("**** error");
  }
  
//  SgFunctionCallExp* fc = isSgFunctionCallExp( exprstatement->get_expression() );
  exprstatement->setAttribute("kaapinotask",(AstAttribute*)1);
}


void Parser::ParseGetLine( std::string& ident )
{
  ident = std::string(rpos);
  rpos = rlast;
}


int Parser::ParseIdentifier( std::string& ident )
{
  char c;
  ident.clear();
  skip_ws();
  c = readchar();
  if ( !isletter(c) )
  {
    putback();
    return c;
  }

  ident.push_back(c);
  do {
    c = readchar();
    if (isletternum(c))
      ident.push_back(c);
    else if (isdigit(c))
      goto digits;
    else {
      putback();
      return c;
    }
  } while ( 1 );

digits:
  ident.push_back(c);
  do {
    c = readchar();
    if (isdigit(c))
      ident.push_back(c);
    else {
      putback();
      return c;
    }
  } while ( 1 );
}


SgUnsignedLongVal* Parser::ParseExprIntegralNumber( )
{
  unsigned long value = 0;
  char c;
  skip_ws();
  c = readchar();
  while (isdigit(c))
  {
    value *= 10;
    value += c - '0';
    c = readchar();
  }
  --rpos;
  return SageBuilder::buildUnsignedLongVal( value );
}

SgVarRefExp* Parser::ParseExprIdentifier( 
    Sg_File_Info* fileInfo, 
    SgScopeStatement* scope 
)
{
  std::string ident;
  ParseIdentifier(ident);
  if (ident.empty())
  {
    std::cerr << "****[kaapi_c2c] Empty identifier."
              << "     In filename '" << fileInfo->get_filename() 
              << "' LINE: " << fileInfo->get_line()
              << std::endl;
    KaapiAbort("**** error");
  }
  return SageBuilder::buildVarRefExp (ident, scope );
}

SgExpression* Parser::ParseExprConstant( 
    Sg_File_Info* fileInfo, 
    SgScopeStatement* scope 
)
{
  char c;
  c = readchar();
  putback();
  if (isdigit(c))
  {
    return ParseExprIntegralNumber( );
  }
  return ParseExprIdentifier(fileInfo,scope);
}

SgExpression* Parser::ParsePrimaryExpression( 
    Sg_File_Info* fileInfo, 
    SgScopeStatement* scope 
)
{
  SgExpression* expr;
  char c;
  skip_ws();
  c = readchar();
  if (c == '(')
  {
    expr = ParseExpression(fileInfo, scope );
    skip_ws();
    c = readchar();
    if (c != ')') 
    {
      std::cerr << "****[kaapi_c2c] Error found '" << c 
                << "'. Missing ')' in primary expression."
                << "     In filename '" << fileInfo->get_filename() 
                << "' LINE: " << fileInfo->get_line()
                << std::endl;
      KaapiAbort("**** error");
    }
    return expr;
  }
  putback();
  expr = ParseExprConstant(fileInfo, scope );
  return expr;
}


/**/
SgExpression* Parser::ParseUnaryExpression( 
    Sg_File_Info* fileInfo, 
    SgScopeStatement* scope 
)
{
  SgExpression* expr;

/* currently do not handle sizeof expression 
   putback seems to be limited to 1 char.
*/
  const char* save_rpos = rpos;
  skip_ws();
  char sizeof_name[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  int nitem = sscanf(rpos, "%7s", sizeof_name);
  if ((nitem !=0) && (strcmp(sizeof_name, "sizeof") == 0))
  {
    char c;
    skip_ws();
    c = readchar();
    if (c == '(')
    {
      std::string type_name;
      ParseIdentifier( type_name );
      SgType* type = SageInterface::lookupNamedTypeInParentScopes(type_name, scope); 
      /* should be a type !!! */
      if (type ==0)
      {
        std::cerr << "****[kaapi_c2c] Error. Unknown type in cast expression.\n"
                  << "     In filename '" << fileInfo->get_filename() 
                  << "' LINE: " << fileInfo->get_line()
                  << std::endl;
        KaapiAbort("**** error");
      }

      skip_ws();
      c = readchar();
      if (c != ')') 
      {
        std::cerr << "****[kaapi_c2c] Error. Missing ')' in cast expression.\n"
                  << "     In filename '" << fileInfo->get_filename() 
                  << "' LINE: " << fileInfo->get_line()
                  << std::endl;
        KaapiAbort("**** error");
      }
      expr = SageBuilder::buildSizeOfOp(type);
      return expr;
    }
    else 
    {
      putback();
      expr = ParseUnaryExpression(fileInfo, scope );
      expr = SageBuilder::buildSizeOfOp(expr);
      return expr;
    }
  }
  else {
    /* putback what has been read */
    rpos = save_rpos;
  }
  
  expr = ParsePrimaryExpression( fileInfo, scope);

  return expr;
}


/**/
SgExpression* Parser::ParseCastExpression( 
    Sg_File_Info* fileInfo, 
    SgScopeStatement* scope 
)
{
  SgExpression* expr;
  char c;
  skip_ws();
  c = readchar();
  if (c == '(') 
  {
    std::string type_name;
    ParseIdentifier( type_name );
    SgType* type = SageInterface::lookupNamedTypeInParentScopes(type_name, scope); 
    /* should be a type !!! */
    if (type ==0)
    {
      std::cerr << "****[kaapi_c2c] Error. Unknown type in cast expression."
                << "     In filename '" << fileInfo->get_filename() 
                << "' LINE: " << fileInfo->get_line()
                << std::endl;
      KaapiAbort("**** error");
    }

    skip_ws();
    c = readchar();
    if (c != ')')
    {
      std::cerr << "****[kaapi_c2c] Error. Missing ')' in cast expression."
                << "     In filename '" << fileInfo->get_filename() 
                << "' LINE: " << fileInfo->get_line()
                << std::endl;
      KaapiAbort("**** error");
    }
    expr = ParseCastExpression(fileInfo, scope);
    expr = SageBuilder::buildCastExp (expr, type );
    return expr;
  }
  putback();
  expr= ParseUnaryExpression( fileInfo, scope);
  return expr;
}


/**/
SgExpression* Parser::ParseMultiplicativeExpression( 
    Sg_File_Info* fileInfo, 
    SgScopeStatement* scope 
)
{
  char c;
  SgExpression* expr= ParseCastExpression( fileInfo, scope);
redo:
  skip_ws();
  c = readchar();
  if (c == '*')
  {
    expr = SageBuilder::buildMultiplyOp( expr, ParseCastExpression( fileInfo, scope) );
    goto redo;
  }
  else if (c == '/')
  {
    expr = SageBuilder::buildDivideOp( expr, ParseCastExpression( fileInfo, scope) );
    goto redo;
  }
  else if (c == '%')
  {
    expr = SageBuilder::buildModOp( expr, ParseCastExpression( fileInfo, scope) );
    goto redo;
  }
  else 
    putback();
  return expr;
}


SgExpression* Parser::ParseAdditiveExpression( 
    Sg_File_Info* fileInfo, 
    SgScopeStatement* scope 
)
{
  char c;
  SgExpression* expr= ParseMultiplicativeExpression( fileInfo, scope);
redo:
  skip_ws();
  c = readchar();
  if (c == '+')
  {
    expr = SageBuilder::buildAddOp( expr, ParseMultiplicativeExpression( fileInfo, scope) );
    goto redo;
  }
  else if (c == '-')
  {
    expr = SageBuilder::buildSubtractOp( expr, ParseMultiplicativeExpression( fileInfo, scope) );
    goto redo;
  }
  else 
    putback();
  return expr;
}


SgExpression* Parser::ParseExpression( 
    Sg_File_Info* fileInfo, 
    SgScopeStatement* scope 
)
{
  SgExpression* expr;
  expr = ParseAdditiveExpression( fileInfo, scope);
  return expr;
}


KaapiStorage_t Parser::ParseStorage( 
    Sg_File_Info*       fileInfo, 
    SgScopeStatement*   scope 
)
{
  std::string name;
  const char* save_rpos = rpos;
  ParseIdentifier( name );
  for (size_t i=0; i<name.size(); ++i)
    name[i] = tolower(name[i]);

  if ((name == "c") || (name == "rowmajor")) return KAAPI_ROW_MAJOR;
  if ((name == "fortran") || (name == "columnmajor")) return KAAPI_COL_MAJOR;

  rpos = save_rpos;
  return KAAPI_BAD_STORAGE;
}


KaapiAccessMode_t Parser::ParseAccessMode( 
    Sg_File_Info* fileInfo
)
{
  std::string name;
  const char* save_rpos = rpos;
  ParseIdentifier( name );
  for (size_t i=0; i<name.size(); ++i)
    name[i] = tolower(name[i]);

  if ((name == "write") || (name == "w") || (name == "output")) return KAAPI_W_MODE;
  if ((name == "read") || (name == "r") || (name == "input")) return KAAPI_R_MODE;
  if ((name == "readwrite") || (name == "rw") || (name == "inout")) return KAAPI_RW_MODE;
  if ((name == "reduction") || (name == "cw")) return KAAPI_CW_MODE;
  if ((name == "value") || (name == "v")) return KAAPI_V_MODE;
  if ((name == "global") || (name == "g")) return KAAPI_GLOBAL_MODE;
  if (name == "highpriority") return KAAPI_HIGHPRIORITY_MODE;

  rpos = save_rpos;
  return KAAPI_VOID_MODE;
}


KaapiReduceOperator_t* Parser::ParseReduceOperator( 
    Sg_File_Info* fileInfo
)
{
  std::string name;
  const char* save_rpos = rpos;
  ParseIdentifier( name );
  
  if (name.empty()) 
  {
    /* try to read char -> basic operator +, - etc */
    char c;
    c = readchar();
    if ( 
         (c == '+') || (c == '-') || (c == '*')
      || (c == '^'))
    {
      name = c;
      c = readchar();
      if (c != ' ')
      {
        rpos= save_rpos;
        return 0;
      }
      putback();
    }
    else if ((c == '&') || (c == '|'))
    { /* may be && or || */
      char c1;
      c1 = readchar();
      if (c != c1)
      {
        putback();
        name = c;
      }
      else {
        if (c == '&') name = "&&";
        else name = "||";
      }
    }
    else {
      name = "ne cherchepas_ca_n'existe pas";
      rpos = save_rpos;
      return 0;
    }
  }
  std::map<std::string,KaapiReduceOperator_t*>::iterator curr =
    kaapi_user_definedoperator.find(name);
  if (curr == kaapi_user_definedoperator.end())
  {
    std::cerr << "****[kaapi_c2c] Error. Unknown reduce operator name '" << name << "'."
              << "     In filename '" << fileInfo->get_filename() 
              << "' LINE: " << fileInfo->get_line()
              << std::endl;
    KaapiAbort("**** error");
  }
  
  return curr->second;
}



KaapiReduceOperator_t* Parser::ParseReductionDeclaration( 
    Sg_File_Info* fileInfo, 
    SgScopeStatement* scope 
)
{
  std::string name;
  char c;
  
  skip_ws();
  c = readchar();
  if (c != '(')
  {
    std::cerr << "****[kaapi_c2c] Error. Missing '(' in declare reduction clause."
              << "     In filename '" << fileInfo->get_filename() 
              << "' LINE: " << fileInfo->get_line()
              << std::endl;
    return 0;
  }
  ParseIdentifier(name);
  if (name.empty())
  {
    std::cerr << "****[kaapi_c2c] Error. Invalid name in declare reduction clause."
              << "     In filename '" << fileInfo->get_filename() 
              << "' LINE: " << fileInfo->get_line()
              << std::endl;
    return 0;
  }
  KaapiReduceOperator_t* newop = new KaapiReduceOperator_t;
  newop->name = name;
  
  skip_ws();
  c = readchar();
  if (c != ':')
  {
    delete newop;
    std::cerr << "****[kaapi_c2c] Error. Missing ':' in declare reduction clause."
              << "     In filename '" << fileInfo->get_filename() 
              << "' LINE: " << fileInfo->get_line()
              << std::endl;
    return 0;
  }
  ParseIdentifier(name);
  if (name.empty())
  {
    delete newop;
    std::cerr << "****[kaapi_c2c] Error. Invalid name in declare reduction clause."
              << "     In filename '" << fileInfo->get_filename() 
              << "' LINE: " << fileInfo->get_line()
              << std::endl;
    return 0;
  }
  /* look up for name as a function 
     - not that due to overloading (C++) a function may appears multiple time.
     - the concrete type is only known during utilization of a variable reduction
     Thus we postpone the verification until the definition of variable in reduciton clause
  */
  newop->name_reducor = name;
  
  skip_ws();
  c = readchar();
  if (c != ')')
  {
    delete newop;
    std::cerr << "****[kaapi_c2c] Error. Missing ')' in declare reduction clause."
              << "     In filename '" << fileInfo->get_filename() 
              << "' LINE: " << fileInfo->get_line()
              << std::endl;
    return 0;
  }
  skip_ws();
  c = readchar();
  if (c == EOF) 
   return newop;
  if (!isletter(c))
  {
    delete newop;
    std::cerr << "****[kaapi_c2c] Error. Waiting for 'identity'  in declare reduction clause."
              << "     In filename '" << fileInfo->get_filename() 
              << "' LINE: " << fileInfo->get_line()
              << std::endl;
    return 0;
  }
  putback();
  ParseIdentifier(name);
  if (name != "identity")
  {
    delete newop;
    std::cerr << "****[kaapi_c2c] Error. Waiting for 'identity'  in declare reduction clause."
              << "     In filename '" << fileInfo->get_filename() 
              << "' LINE: " << fileInfo->get_line()
              << std::endl;
    return 0;
  }
  
  skip_ws();
  c = readchar();
  if (c != '(')
  {
    std::cerr << "****[kaapi_c2c] Error. Missing '(' in declare reduction clause."
              << "     In filename '" << fileInfo->get_filename() 
              << "' LINE: " << fileInfo->get_line()
              << std::endl;
    return 0;
  }
  
  ParseIdentifier(name);
  if (name.empty())
  {
    delete newop;
    std::cerr << "****[kaapi_c2c] Error. Invalid name for identity function in declare reduction clause."
              << "     In filename '" << fileInfo->get_filename() 
              << "' LINE: " << fileInfo->get_line()
              << std::endl;
    return 0;
  }
  newop->name_redinit = name;
  
  skip_ws();
  c = readchar();
  if (c != ')')
  {
    delete newop;
    std::cerr << "****[kaapi_c2c] Error. Missing ')' in declare reduction clause."
              << "     In filename '" << fileInfo->get_filename() 
              << "' LINE: " << fileInfo->get_line()
              << std::endl;
    return 0;
  }
  return newop;
}


void Parser::ParseDimension( 
    Sg_File_Info* fileInfo, 
    KaapiParamAttribute* kpa, 
    SgScopeStatement* scope 
)
{
  char c;

  skip_ws();
  c = readchar();
  if (c != '[') 
  {
    std::cerr << "****[kaapi_c2c] Error. Missing '[' in dimension expression."
              << "     In filename '" << fileInfo->get_filename() 
              << "' LINE: " << fileInfo->get_line()
              << std::endl;
    KaapiAbort("**** error");
  }
  kpa->ndim[kpa->dim] = ParseExpression( fileInfo, scope );
  if (kpa->ndim[kpa->dim] ==0)
  {
    std::cerr << "****[kaapi_c2c] Error. BAd expression in dimension."
              << "     In filename '" << fileInfo->get_filename() 
              << "' LINE: " << fileInfo->get_line()
              << std::endl;
    KaapiAbort("**** error");
  }
  
  skip_ws();
  c = readchar();
  if (c != ']') 
  {
    std::cerr << "****[kaapi_c2c] Error. Missing ']' in dimension expression."
              << "     In filename '" << fileInfo->get_filename() 
              << "' LINE: " << fileInfo->get_line()
              << std::endl;
    KaapiAbort("**** error");
  }
  if (++kpa->dim == 3) 
  {
    std::cerr << "****[kaapi_c2c] Error. Too high dimension for array (hard limit=2)."
              << "     In filename '" << fileInfo->get_filename() 
              << "' LINE: " << fileInfo->get_line()
              << std::endl;
    KaapiAbort("**** error");
  }
}


void Parser::ParseNDimensions( 
    Sg_File_Info* fileInfo, 
    KaapiParamAttribute* kpa, 
    SgScopeStatement* scope 
)
{
  char c;

next_dimension:
  skip_ws();
  c = readchar();
  if (c == '[')
  {
    putback();
    ParseDimension( fileInfo, kpa, scope );
    goto next_dimension;
  }
  putback();
  return;
}


void Parser::ParseComplexView( 
    Sg_File_Info*        fileInfo, 
    KaapiParamAttribute* kpa, 
    SgScopeStatement*    scope 
)
{
  std::string name;
  char c;
  const char* save_rpos;
  
  skip_ws();
  c = readchar();
  if (c != '{')
  {
    putback();
    return;
  }
  
redo:
  skip_ws();
  save_rpos = rpos;
  ParseIdentifier( name );
  
  if ((name == "storage") || (name == "ld") || (name == "lda"))
  {
    if (name == "lda")
    {
      std::cerr << "****[kaapi_c2c] warning. Deprecated use of 'lda'"
                << "     In filename '" << fileInfo->get_filename() 
                << "' LINE: " << fileInfo->get_line()
                << std::endl;
    }
    skip_ws();
    c = readchar();
    if (c != '=') 
    {
      std::cerr << "****[kaapi_c2c] Error. Missing '=' after '" << name << "'."
                << "     In filename '" << fileInfo->get_filename() 
                << "' LINE: " << fileInfo->get_line()
                << std::endl;
      KaapiAbort("**** error");
    }
    if (name =="storage")
    {
      if (kpa->storage != KAAPI_BAD_STORAGE) 
      {
        std::cerr << "****[kaapi_c2c] Error. Storage attribut already defined for same identifier."
                  << "     In filename '" << fileInfo->get_filename() 
                  << "' LINE: " << fileInfo->get_line()
                  << std::endl;
        KaapiAbort("**** error");
      }
      kpa->storage = ParseStorage( fileInfo, scope );
      if (kpa->storage != KAAPI_BAD_STORAGE) 
      {
        std::cerr << "****[kaapi_c2c] Error. Bad storage name: waiting 'C', 'RowMajor', 'Fortran' or 'ColumnMajor'."
                  << "     In filename '" << fileInfo->get_filename() 
                  << "' LINE: " << fileInfo->get_line()
                  << std::endl;
        KaapiAbort("**** error");
      }
    }
    else 
    { /* name == ld */
      if (kpa->lda != 0) 
      {
        std::cerr << "****[kaapi_c2c] Error. LDA attribut already defined for same identifier."
                  << "     In filename '" << fileInfo->get_filename() 
                  << "' LINE: " << fileInfo->get_line()
                  << std::endl;
        KaapiAbort("**** error");
      }
      kpa->lda = ParseExprIdentifier(fileInfo,scope);
      if (kpa->lda == 0) 
      {
        std::cerr << "****[kaapi_c2c] Error. LDA attribut, waiting identifier."
                  << "     In filename '" << fileInfo->get_filename() 
                  << "' LINE: " << fileInfo->get_line()
                  << std::endl;
        KaapiAbort("**** error");
      }
    }
    
    /* here: c==';' or '}' */
    skip_ws();
    c = readchar();
    if (c == ';') goto redo; /* next attribut */
    if (c == '}') return;
    std::cerr << "****[kaapi_c2c] Error. Missing ';' or '}' in complex view definition. Found: '" << c << "'."
              << "     In filename '" << fileInfo->get_filename() 
              << "' LINE: " << fileInfo->get_line()
              << std::endl;
    KaapiAbort("**** error");
  }
  else {
    rpos = save_rpos; /* putback token */
  }

  ParseNDimensions( fileInfo, kpa, scope );
  skip_ws();
  c = readchar();
  if (c == ';') goto redo;
  if (c == '}') return;
  
  std::cerr << "****[kaapi_c2c] Error. Missing ';' or '}' in complex view definition. Found: '" << c << "'."
            << "     In filename '" << fileInfo->get_filename() 
            << "' LINE: " << fileInfo->get_line()
            << std::endl;
  KaapiAbort("**** error");
}

void Parser::ParseRangeDeclaration( 
    Sg_File_Info*        fileInfo, 
    KaapiParamAttribute* kpa, 
    SgScopeStatement*    scope 
)
{
  std::string name;
  char c;

  skip_ws();
  c = readchar();
  if (c == '[')
  {
    SgVarRefExp* expr_first_bound;
    SgVarRefExp* expr_second_bound;

    /* range 1D definition */
    expr_first_bound = ParseExprIdentifier( fileInfo, scope );
    /* parse .. */
    skip_ws();
    c= readchar();
    if (c != ':')
    {
      std::cerr << "****[kaapi_c2c] Error. Missing ':' for range declaration. Found'" << c << "'."
                << "     In filename '" << fileInfo->get_filename() 
                << "' LINE: " << fileInfo->get_line()
                << std::endl;
      KaapiAbort("**** error");
    }
    expr_second_bound = ParseExprIdentifier( fileInfo, scope );

    skip_ws();
    c= readchar();
    /* parse ')' or ']' */
    if ( (c != ')') && ( c != ']'))
    {
      std::cerr << "****[kaapi_c2c] Error. Missing ')' or ']' for end of range declaration. Found'" << c << "'."
                << "     In filename '" << fileInfo->get_filename() 
                << "' LINE: " << fileInfo->get_line()
                << std::endl;
      KaapiAbort("**** error");
    }
    
    if (c == ')') /* open range declaration */
      kpa->type = KAAPI_OPEN_RANGE_TYPE;
    if (c == ']') /* closed range declaration */
      kpa->type = KAAPI_CLOSE_RANGE_TYPE;
    kpa->expr_firstbound  = expr_first_bound;
    kpa->first_bound = expr_first_bound->get_symbol()->get_name().str();
    kpa->expr_secondbound = expr_second_bound;
    kpa->second_bound = expr_second_bound->get_symbol()->get_name().str();
  }
  else {
    putback();
    ParseIdentifier( name );
    if (name == "")
    {
      std::cerr << "****[kaapi_c2c] Error. Missing identifier for range declaration."
                << "     In filename '" << fileInfo->get_filename() 
                << "' LINE: " << fileInfo->get_line()
                << std::endl;
      KaapiAbort("**** error");
    }
    kpa->type = KAAPI_ARRAY_NDIM_TYPE;
    kpa->name = strdup(name.c_str());
    skip_ws();
    c = readchar();
    if (c == '{')
    {
      putback();
      ParseComplexView( fileInfo, kpa, scope );
      return;
    }
    else if (c == '[')
    {
      putback();
      ParseNDimensions( fileInfo, kpa, scope );
      return;
    }
    else if ((c == ')') || (c == ','))
    {
      kpa->dim = 0; /* means dim=1, but single element */
      putback();
      return;
    }
    std::cerr << "****[kaapi_c2c] Error. Missing end ')' or ','."
              << "     In filename '" << fileInfo->get_filename() 
              << "' LINE: " << fileInfo->get_line()
              << std::endl;
    KaapiAbort("**** error");
  }
}


/* Parse the list of formal parameter definitions      
*/
void Parser::ParseListParamDecl( 
    Sg_File_Info*       fileInfo, 
    KaapiTaskAttribute* kta,
    KaapiAccessMode_t   mode,
    SgScopeStatement*   scope 
)
{
  char c;
  KaapiParamAttribute* kpa;
  KaapiReduceOperator_t* redop = 0;

  if (mode == KAAPI_CW_MODE)
  {
    /* parse operator */
    redop = ParseReduceOperator(fileInfo );
    if (redop ==0)
    {
      std::cerr << "****[kaapi_c2c] Error. Unknown reduction operator."
                << "     In filename '" << fileInfo->get_filename() 
                << "' LINE: " << fileInfo->get_line()
                << std::endl;
      KaapiAbort("**** error");
    }
    
    /* parse : */
    skip_ws();
    c = readchar();
    if (c != ':')
    {
      std::cerr << "****[kaapi_c2c] Error. Missing ':' in reduction clause"
                << "     In filename '" << fileInfo->get_filename() 
                << "' LINE: " << fileInfo->get_line()
                << std::endl;
      KaapiAbort("**** error");
    }
  }
    
redo:
  kpa = new KaapiParamAttribute;
  ParseRangeDeclaration( fileInfo, kpa, scope );
  switch (kpa->type)
  {
    case KAAPI_ARRAY_NDIM_TYPE:
    {
      if (mode != KAAPI_GLOBAL_MODE)
      {
        std::map<std::string,int>::iterator curr = kta->lookup.find( kpa->name );
        if (curr == kta->lookup.end())
        {
          std::cerr << "****[kaapi_c2c] Error. Unkown formal parameter identifier in declaration '" << kpa->name << "'."
                    << "     In filename '" << fileInfo->get_filename() 
                    << "' LINE: " << fileInfo->get_line()
                    << std::endl;
          KaapiAbort("**** error");
        }
        int ith = curr->second;
        kta->formal_param[ith].mode = mode;
        kta->formal_param[ith].attr = kpa;
        kta->formal_param[ith].redop = redop;
        kta->israngedecl[ith]       = 0;
      }
      else { /* global variable */
        SgVariableSymbol* var = SageInterface::lookupVariableSymbolInParentScopes( kpa->name, scope );
        if (var ==0)
        {
          std::cerr << "****[kaapi_c2c] Error. Unkown global variable identifier '" << kpa->name << "'."
                    << "     In filename '" << fileInfo->get_filename() 
                    << "' LINE: " << fileInfo->get_line()
                    << std::endl;
          KaapiAbort("**** error");
        }
        KaapiTaskFormalParam kpa_extra;
        kpa_extra.mode     = mode;
        kpa_extra.initname = var->get_declaration();
        kpa_extra.type     = var->get_type();
        kta->extra_param.push_back( kpa_extra );
      }

    } break;
    
    case KAAPI_OPEN_RANGE_TYPE:
    case KAAPI_CLOSE_RANGE_TYPE:
    {
      std::map<std::string,int>::iterator curr1 = kta->lookup.find( kpa->first_bound );
      if (curr1 == kta->lookup.end())
      {
        std::cerr << "****[kaapi_c2c] Error. Unkown formal parameter identifier in range declaration '" << kpa->first_bound << "'."
                  << "     In filename '" << fileInfo->get_filename() 
                  << "' LINE: " << fileInfo->get_line()
                  << std::endl;
        KaapiAbort("**** error");
      }
      std::map<std::string,int>::iterator curr2 = kta->lookup.find( kpa->second_bound );
      if (curr2 == kta->lookup.end())
      {
        std::cerr << "****[kaapi_c2c] Error. Unkown formal parameter identifier in range declaration '" << kpa->second_bound << "'."
                  << "     In filename '" << fileInfo->get_filename() 
                  << "' LINE: " << fileInfo->get_line()
                  << std::endl;
        KaapiAbort("**** error");
      }
      kpa->index_firstbound  = curr1->second;
      kpa->index_secondbound = curr2->second;
      kta->formal_param[curr1->second].mode  = mode;
      kta->formal_param[curr1->second].attr  = kpa;
      kta->israngedecl[curr1->second]        = 1;
      kta->formal_param[curr1->second].redop = redop;

      kta->formal_param[curr2->second].mode  = mode;
      kta->formal_param[curr2->second].attr  = kpa;
      kta->israngedecl[curr2->second]        = 2;
      kta->formal_param[curr2->second].redop = redop;
    } break;

    
    case KAAPI_BAD_PARAM_TYPE:
    default:
      std::cerr << "****[kaapi_c2c] Error. Bad list of declaration."
                << "     In filename '" << fileInfo->get_filename() 
                << "' LINE: " << fileInfo->get_line()
                << std::endl;
      KaapiAbort("**** error");
  }
  
  skip_ws();
  c = readchar();
  if (c == ',') goto redo;
  if (c == ')') 
  {
    putback();
    return;
  }

  std::cerr << "****[kaapi_c2c] Error. Missing ',' or ')' at the end of list of declaration."
            << "     In filename '" << fileInfo->get_filename() 
            << "' LINE: " << fileInfo->get_line()
            << std::endl;
  KaapiAbort("**** error");
}


void Parser::ParseListParam( 
    Sg_File_Info*       fileInfo, 
    KaapiTaskAttribute* kta,
    SgScopeStatement*   scope 
)
{
  char c;
  KaapiAccessMode_t mode;

redo:
  mode = ParseAccessMode( fileInfo );
  
  if (mode == KAAPI_VOID_MODE) goto next_clause;
  
  skip_ws();
  c = readchar();
  if (c == EOF) return;
  
  if (mode == KAAPI_HIGHPRIORITY_MODE) 
  { /* ignore the mode */
    goto redo;
  }
  
  if (c != '(')
  {
    std::cerr << "****[kaapi_c2c] Error. Missing '(' in access mode list."
              << "     In filename '" << fileInfo->get_filename() 
              << "' LINE: " << fileInfo->get_line()
              << std::endl;
    KaapiAbort("**** error");
  }

  ParseListParamDecl( fileInfo, kta, mode, scope );
  
  skip_ws();
  c = readchar();
  if (c != ')')
  {
    std::cerr << "****[kaapi_c2c] Error. Missing ')' at the end of access mode list."
              << "     In filename '" << fileInfo->get_filename() 
              << "' LINE: " << fileInfo->get_line()
              << std::endl;
    KaapiAbort("**** error");
  }

 next_clause:  
  c = readchar();
  if (c == EOF) return;
  goto redo;
}


/** #pragma kaapi data parsing
*/

/* Attribut attached to a #pragma kaapi data alloca variable declaration */
class KaapiDataAttribute : public AstAttribute {
public:
  KaapiDataAttribute ( SgVariableSymbol* newsymbol ) 
   : symbol(newsymbol)
  { }

  SgVariableSymbol* symbol;
};

class nestedVarRefVisitorTraversal : public AstSimpleProcessing {
public:
  nestedVarRefVisitorTraversal( )
  { }
  virtual void visit(SgNode* n)
  {
    SgVarRefExp* varref = isSgVarRefExp(n);
    if (varref != 0) 
    {
      SgVariableSymbol* varsym = varref->get_symbol();
      KaapiDataAttribute* kada = (KaapiDataAttribute*)varsym->getAttribute("kaapidata");
      if (kada !=0)
      {
  #if 0
#if CONFIG_ENABLE_DEBUG
        Sg_File_Info* fileInfo = varref->get_file_info();
        std::cerr <<  "In filename '" << fileInfo->get_filename() << "' LINE: " << fileInfo->get_line()
                  << " Found use of VarRef for variable name:" << varsym->get_name().str()
                  << std::endl;
#endif // CONFIG_ENABLE_DEBUG
  #endif
        
        SgVarRefExp*       newvarref = SageBuilder::buildVarRefExp (kada->symbol );
        SgPointerDerefExp* deref = SageBuilder::buildPointerDerefExp( newvarref );
        
        /* replace the expression */
        SgExpression* parentexptr = isSgExpression(varref->get_parent());
        parentexptr->replace_expression( varref, deref );
      }
    }    
  }
};

void Parser::DoKaapiPragmaData( SgNode* node )
{
  std::string name;
  char c;
  SgPragmaDeclaration* sgp = isSgPragmaDeclaration(node);
  
  ParseIdentifier( name );
  if (name != "alloca")
  {
    Sg_File_Info* fileInfo = sgp->get_file_info();
    std::cerr <<  "In filename '" << fileInfo->get_filename() << "' LINE: " << fileInfo->get_line()
              << " #pragma kaapi data: expecting 'alloca' clause, found '" << name << "'"
              << std::endl;
    KaapiAbort("**** error");
  }

//  SgBasicBlock* bbnode = isSgBasicBlock(sgp->get_parent());
  SgScopeStatement* bbnode = SageInterface::getScope(sgp);
  if (bbnode ==0)
  {
    Sg_File_Info* fileInfo = sgp->get_file_info();
    std::cerr <<  "In filename '" << fileInfo->get_filename() << "' LINE: " << fileInfo->get_line()
              << " #pragma kaapi data: invalid scope declaration"
              << std::endl;
    KaapiAbort("**** error");
  }

  SgSymbolTable* symtable = bbnode->get_symbol_table();
  
  /* parse the variables declaration in the pragma string */
  skip_ws();
  c = readchar();
  if (c != '(') 
  {
    Sg_File_Info* fileInfo = sgp->get_file_info();
    std::cerr <<  "In filename '" << fileInfo->get_filename() << "' LINE: " << fileInfo->get_line()
              << " #pragma kaapi data alloca clause: missing '('"
              << std::endl;
    KaapiAbort("**** error");
  }
  
  bool findend = false;
  while (1)
  {
    skip_ws();
    ParseIdentifier( name );
    if (name.size() >0)
    {
#if 0
      std::cout << "-- Look for variable: '" << name << "' separator:'" << c << "'" << std::endl;
#endif
      
      SgName varname (name);
      SgVariableSymbol* var = symtable->find_variable( varname );
      if (var == 0)
      {
        Sg_File_Info* fileInfo = sgp->get_file_info();
        std::cerr <<  "In filename '" << fileInfo->get_filename() << "' LINE: " << fileInfo->get_line()
                  << " #pragma kaapi data alloca clause: variable name '" << name << "' not found in the scope"
                  << std::endl;
        KaapiAbort("**** error");
      }

      /* add a variable with name _kaapi_varname and type the pointer of the current type */
      SgType* type = var->get_type();
      SgPointerType* newtype = SageBuilder::buildPointerType(type);
      
      /* if it is not alread done: add __kaapi_thread variable 
         in the top scope:
      */
      SgVariableSymbol* newkaapi_threadvar = 
          SageInterface::lookupVariableSymbolInParentScopes(
              "__kaapi_thread", 
              bbnode 
      );
      if (newkaapi_threadvar ==0)
        buildInsertDeclarationKaapiThread( bbnode);
  
      SgVariableDeclaration* newvardecl = SageBuilder::buildVariableDeclaration ( 
        "_kaapi_"+name, 
        newtype, 
        SageBuilder::buildAssignInitializer(
          SageBuilder::buildCastExp(
            SageBuilder::buildFunctionCallExp(
              "kaapi_alloca",
              SageBuilder::buildPointerType(SageBuilder::buildVoidType()),
              SageBuilder::buildExprListExp(
                SageBuilder::buildVarRefExp ("__kaapi_thread", bbnode ),
                SageBuilder::buildSizeOfOp( type )
              ),
              bbnode
            ),
            newtype 
          ),
          0
        ),
        bbnode
      );
      SgVariableSymbol* newvar = symtable->find_variable( "_kaapi_"+name );

      var->setAttribute("kaapidata", 
        new KaapiDataAttribute( newvar )
      );
      SageInterface::insertStatement(sgp, newvardecl, false);
      SageInterface::removeStatement( var->get_declaration ()->get_declaration() );
    }
    else {
        Sg_File_Info* fileInfo = sgp->get_file_info();
        std::cerr <<  "In filename '" << fileInfo->get_filename() << "' LINE: " << fileInfo->get_line()
                  << " #pragma kaapi data alloca clause: mal formed variable list, missing variable name ?"
                  << std::endl;
        KaapiAbort("**** error");
    }
    skip_ws();
    c = readchar();
    if (c == ')') 
    {
      findend = true;
      break;
    }
    if ((c != ' ') && (c != ','))
    {
        Sg_File_Info* fileInfo = sgp->get_file_info();
        std::cerr <<  "In filename '" << fileInfo->get_filename() << "' LINE: " << fileInfo->get_line()
                  << " #pragma kaapi data alloca clause: unrecognized character '" << c << "'"
                  << std::endl;
        KaapiAbort("**** error");
    }
    if (c == EOF)
      break;
  }
  if (!findend) 
  {
    Sg_File_Info* fileInfo = sgp->get_file_info();
    std::cerr <<  "In filename '" << fileInfo->get_filename() << "' LINE: " << fileInfo->get_line()
              << " #pragma kaapi data alloca clause: missing ')'"
              << std::endl;
    KaapiAbort("**** error");
  }

  /* replace #pragma kaapi data alloca by the extern definition of kaapi_alloca(size_t) */
  //  SageInterface::prependStatement(decl_alloca, bbnode);
//  SageInterface::removeStatement(sgp);
  
  /* traverse all the expression inside the bbnode to replace use of variables */
  nestedVarRefVisitorTraversal replace;
  replace.traverse(bbnode,postorder);
}


/** #pragma kaapi barrier parsing
*/
void Parser::DoKaapiPragmaBarrier( SgPragmaDeclaration* sgp )
{
  std::string name;

  SgScopeStatement* bbnode = isSgBasicBlock(SageInterface::getScope(sgp));
  if (bbnode ==0)
  {
    Sg_File_Info* fileInfo = sgp->get_file_info();
    std::cerr <<  "In filename '" << fileInfo->get_filename() << "' LINE: " << fileInfo->get_line()
              << " #pragma kaapi sync: invalid scope declaration"
              << std::endl;
    KaapiAbort("**** error");
  }

  SgExprStatement* callStmt = SageBuilder::buildFunctionCallStmt
  (    "kaapi_sched_sync", 
       SageBuilder::buildIntType(), 
       SageBuilder::buildExprListExp(),
       bbnode
  );
  /* insert after */
  SageInterface::insertStatement(sgp,callStmt,false);
//  SageInterface::replaceStatement(sgp,callStmt);
}


/** #pragma kaapi waiton parsing
*/
void Parser::DoKaapiPragmaWaiton( SgPragmaDeclaration* sgp )
{
  std::string name;

  SgBasicBlock* bbnode = isSgBasicBlock(sgp->get_parent());
  if (bbnode ==0)
  {
    Sg_File_Info* fileInfo = sgp->get_file_info();
    std::cerr <<  "In filename '" << fileInfo->get_filename() << "' LINE: " << fileInfo->get_line()
              << " #pragma kaapi waiton: invalid scope declaration"
              << std::endl;
    KaapiAbort("**** error");
  }
//  SgNode* nextnode = 
          sgp->get_parent()-> get_traversalSuccessorByIndex( 
              sgp->get_parent()->get_childIndex( sgp ) + 1);

  SgExprStatement* callStmt = SageBuilder::buildFunctionCallStmt
  (    "kaapi_sched_sync", 
       SageBuilder::buildIntType(), 
       SageBuilder::buildExprListExp(),
       bbnode
  );

  /* Insert after */
  SageInterface::insertStatement(sgp,callStmt,false);
//  SageInterface::replaceStatement(sgp,callStmt);

// TODO: parse list of variables 
}


/** #pragma kaapi parallel
    TODO: parse num_threads(num)
*/
void Parser::DoKaapiPragmaParallelRegion( SgPragmaDeclaration* sgp )
{
  std::string name;
  const char* save_rpos;
  int flagnowait =0;
  
  SgBasicBlock* bbnode = isSgBasicBlock(sgp->get_parent());
  if (bbnode ==0)
  {
    Sg_File_Info* fileInfo = sgp->get_file_info();
    std::cerr <<  "In filename '" << fileInfo->get_filename() << "' LINE: " << fileInfo->get_line()
              << " #pragma kaapi parallel: invalid scope declaration"
              << std::endl;
    KaapiAbort("**** error");
  }

  /* here 2 cases: #pragma kaapi parallel or #pragma kaapi loop
  */
  skip_ws();
  save_rpos = rpos;
  ParseIdentifier( name );

  /* Case of adaptive loop 
  */
  if (name == "loop")
  {
    DoKaapiPragmaLoop( sgp );
    return;
  }
  else {
    rpos = save_rpos;
  }
  skip_ws();
  save_rpos = rpos;
  ParseIdentifier( name );
  if (name == "nowait")
    flagnowait = 1;

  SgNode* nextnode = 
          sgp->get_parent()-> get_traversalSuccessorByIndex( 
              sgp->get_parent()->get_childIndex( sgp ) + 1);

  SgExprStatement* callinitStmt;
  SgExprStatement* callfinishStmt;
  if (isSgStatement(nextnode) !=0)
  {
    /* add a new fictif basicblock that contains only the Statement nextnode */
    SgStatement* nextstmt = isSgStatement(nextnode);
    SgBasicBlock* newbb = SageBuilder::buildBasicBlock();
    SageInterface::insertStatement( nextstmt, newbb, false );
    SageInterface::removeStatement( nextstmt );
    SageInterface::appendStatement( nextstmt, newbb );
    nextnode = newbb;
  }
  if (isSgBasicBlock(nextnode))
  {
    SgStatement* first_stmt;
    SgStatement* last_stmt;
    callinitStmt = SageBuilder::buildFunctionCallStmt
    (    "kaapi_begin_parallel", 
         SageBuilder::buildVoidType(), 
         SageBuilder::buildExprListExp(SageBuilder::buildIntVal(0) ),
         bbnode
    );
    callfinishStmt = SageBuilder::buildFunctionCallStmt
    (    "kaapi_end_parallel", 
         SageBuilder::buildVoidType(), 
         SageBuilder::buildExprListExp( SageBuilder::buildIntVal(flagnowait) ),
         bbnode
    );

    nextnode->setAttribute("kaapiisparallelregion", (AstAttribute*)-1);
    nextnode->setAttribute("kaapiparallelregion", (AstAttribute*)callinitStmt);
    first_stmt = SageInterface::getFirstStatement(isSgBasicBlock(nextnode), true);
    last_stmt = SageInterface::getLastStatement(isSgBasicBlock(nextnode));

    SageInterface::insertStatement( first_stmt, callinitStmt, true );
    SageInterface::insertStatement( last_stmt, callfinishStmt, false );
  } 
  else {
    Sg_File_Info* fileInfo = sgp->get_file_info();
    std::cerr << "****[kaapi_c2c] invalid scope for #pragma kaapi parallel."
              << "     In filename '" << fileInfo->get_filename() 
              << "' LINE: " << fileInfo->get_line()
              << std::endl;
    KaapiAbort("**** error");
  }
}

/** #pragma kaapi init/finish: flag == true if init
*/
void Parser::DoKaapiPragmaInit( SgPragmaDeclaration* sgp, bool flag )
{
  std::string name;

  SgBasicBlock* bbnode = isSgBasicBlock(sgp->get_parent());
  if (bbnode ==0)
  {
    Sg_File_Info* fileInfo = sgp->get_file_info();
    std::cerr <<  "In filename '" << fileInfo->get_filename() << "' LINE: " << fileInfo->get_line()
              << " #pragma kaapi waiton: invalid scope declaration"
              << std::endl;
    KaapiAbort("**** error");
  }
  bbnode->setAttribute("kaapiisparallelregion", (AstAttribute*)-1);
//  SgNode* nextnode = 
          sgp->get_parent()-> get_traversalSuccessorByIndex( 
              sgp->get_parent()->get_childIndex( sgp ) + 1);

  SgExprStatement* callStmt;
  
  if (flag)
  {
    callStmt = SageBuilder::buildFunctionCallStmt
    (    "kaapi_init", 
         SageBuilder::buildVoidType(), 
         SageBuilder::buildExprListExp(
           SageBuilder::buildIntVal (0),
           SageBuilder::buildIntVal (0),
           SageBuilder::buildIntVal (0)
         ),
         bbnode
    );
  }
  else {
    callStmt = SageBuilder::buildFunctionCallStmt
    (    "kaapi_finalize", 
         SageBuilder::buildVoidType(), 
         SageBuilder::buildExprListExp(
         ),
         bbnode
    );
  }
  /* Insert after */
  SageInterface::insertStatement(sgp,callStmt,false);
}


void Parser::DoKaapiPragmaDeclare( SgPragmaDeclaration* sgp )
{
  SgScopeStatement* scope = SageInterface::getScope(sgp);
  Sg_File_Info* fileInfo = sgp->get_file_info();
  
  const char* save_rpos = rpos;
  std::string name;
  ParseIdentifier(name);
  if (name == "reduction")
  {
    KaapiReduceOperator_t* redop = ParseReductionDeclaration(
        fileInfo, 
        scope 
    );
    if (redop ==0)
    {
      KaapiAbort("**** error");
    }
    kaapi_user_definedoperator.insert( std::make_pair( redop->name, redop  ) );

#if CONFIG_ENABLE_DEBUG
    std::cout << "Found declaration of reduction operator:"
              << redop->name
              << " freduce=" << redop->name_reducor
              << " finit=" << redop->name_redinit
              << std::endl;
#endif // CONFIG_ENABLE_DEBUG
    return;
  }
  else {
    std::cerr << "****[kaapi_c2c] #pragma kaapi declare '" << name << "' clause."
              << "     In filename '" << fileInfo->get_filename() 
              << "' LINE: " << fileInfo->get_line()
              << std::endl;
    KaapiAbort("**** error");
    rpos = save_rpos;
    return;
  }

  return;
}
