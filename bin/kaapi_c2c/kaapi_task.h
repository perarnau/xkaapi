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


#ifndef KAAPI_TASK_H_INCLUDED
# define KAAPI_TASK_H_INCLUDED


#include <string>
#include <map>
#include <set>
#include <vector>
#include <sstream>
#include <sys/types.h>
#include "rose_headers.h"


/* link a kaapi task AST function declaration to the pragma kaapi task */
enum KaapiAccessMode_t {
  KAAPI_VOID_MODE = 0,
  KAAPI_W_MODE,
  KAAPI_R_MODE,
  KAAPI_RW_MODE,
  KAAPI_CW_MODE,
  KAAPI_V_MODE,
  KAAPI_GLOBAL_MODE,
  
  KAAPI_HIGHPRIORITY_MODE /* not really a mode. Only used to parse css program */
};


enum KaapiOperator_t {
  KAAPI_NO_OPERATOR = 0,
  KAAPI_ADD_OPERATOR,
  KAAPI_SUB_OPERATOR,
  KAAPI_MUL_OPERATOR,
  KAAPI_AND_OPERATOR,
  KAAPI_OR_OPERATOR,
  KAAPI_XOR_OPERATOR,
  KAAPI_LAND_OPERATOR, /* logical and */
  KAAPI_LOR_OPERATOR, /* logical and */
  
  KAAPI_USER_OPERATOR
};

struct KaapiReduceOperator_t {
  KaapiReduceOperator_t() 
   : isbuiltin(false), name(), name_reducor(), name_redinit()
  {}
  KaapiReduceOperator_t(const char* n, const char* reducor, const char* redinit) 
   : isbuiltin(true), name(n), name_reducor(reducor), name_redinit(redinit)
  {}
  bool  isbuiltin;
  std::string name;
  std::string name_reducor;
  std::string name_redinit;
};
static std::map<std::string,KaapiReduceOperator_t*> kaapi_user_definedoperator;

enum KaapiStorage_t {
  KAAPI_BAD_STORAGE = 0,
  KAAPI_ROW_MAJOR = 1,
  KAAPI_COL_MAJOR = 2
};

enum KaapiParamAttribute_t {
  KAAPI_BAD_PARAM_TYPE = 0,
  KAAPI_ARRAY_NDIM_TYPE = 1,
  KAAPI_OPEN_RANGE_TYPE = 2,
  KAAPI_CLOSE_RANGE_TYPE = 3
};


/* Store view of each parameter */
class KaapiParamAttribute : public AstAttribute {
public:
  KaapiParamAttribute ()
   : type(KAAPI_BAD_PARAM_TYPE),  storage(KAAPI_ROW_MAJOR), lda(0), dim(0) 
  { 
    memset(ndim, 0, sizeof(ndim) );
  }
  KaapiParamAttribute_t  type;
  union {
    struct {
      const char*           name;
      KaapiStorage_t        storage;
      SgExpression*         lda;
      size_t                dim;
      SgExpression*         ndim[3];
    };
    struct {
      const char*           first_bound;
      SgExpression*         expr_firstbound;
      int                   index_firstbound;
      const char*           second_bound;
      SgExpression*         expr_secondbound;
      int                   index_secondbound;
    };
  };
};

struct KaapiTaskFormalParam {
  KaapiTaskFormalParam()
   : mode(KAAPI_VOID_MODE), attr(0), initname(0), type(0), kaapi_format()
  {}
  KaapiAccessMode_t      mode;
  KaapiReduceOperator_t* redop;
  KaapiParamAttribute*   attr;
  SgInitializedName*     initname;
  SgType*                type;
  std::string            kaapi_format; 
};


/**
*/
class KaapiTaskAttribute : public AstAttribute {
public:
  KaapiTaskAttribute () 
    : is_signature(false), 
      has_retval(false), 
      has_this(false),
      paramclass(0),
      typedefparamclass(0),
      func_decl(0),
      wrapper_decl(0),
      fwd_wrapper_decl(0),
      class_decl(0),
      reducer_decl(0)
  { }

  bool				                is_signature;      /* signature pragma */
  std::string                       mangled_name;      /* internal name of the task */
  bool                              has_retval;
  KaapiTaskFormalParam              retval;
  bool                              has_this;
  KaapiTaskFormalParam              thisval;
  std::vector<KaapiTaskFormalParam> formal_param;
  std::vector<KaapiTaskFormalParam> extra_param;
  std::vector<int>                  israngedecl;       /* for range declaration: 0 no, 1:begin, 2: end*/
  std::map<std::string,int>         lookup;
  SgScopeStatement                  scope;             /* for the formal parameter */
  std::string                       name_paramclass;   /* name of the type for arguments */
  SgClassDeclaration*               paramclass;        /* the class declaration for the parameters */
  SgTypedefDeclaration*             typedefparamclass; /* */
  SgFunctionDeclaration*            func_decl;         /* the function declaration */

  std::string                       name_wrapper;      /* name of the DFG wrapper */
  SgFunctionDeclaration*            wrapper_decl;      /* its declaration */
  SgFunctionDeclaration*            fwd_wrapper_decl;      /* its declaration */

  std::string                       name_format;       /* name of the format */

  SgClassDeclaration*		    class_decl;
  SgFunctionDeclaration*	    reducer_decl;

  bool hasReduction() const;
  void buildReductionSet(std::set<SgVariableSymbol*>& symbol_set);
  void buildReductionSet(std::set<KaapiTaskFormalParam*>& param_set);

  SgClassDeclaration* buildInsertClassDeclaration
  (
   SgGlobal* global_scope,
   SgScopeStatement* local_scope,
   SgClassDefinition* class_def = NULL
  );

  // TODO: should be non member static function
  SgFunctionDeclaration* buildInsertReducer
  (SgType* work_type, SgType* result_type, SgGlobal* global_scope);
};


void DoKaapiGenerateFormat( std::ostream& fout, KaapiTaskAttribute* kta);


#endif // ! KAAPI_TASK_H_INCLUDED
