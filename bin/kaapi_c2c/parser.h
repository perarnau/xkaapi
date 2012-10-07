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


#ifndef KAAPI_PARSER_H_INCLUDED
# define KAAPI_PARSER_H_INCLUDED


#include <string.h>
#include <string>
#include <stdexcept>
#include "rose_headers.h"
#include "kaapi_c2c_task.h"


class Parser {
  const char* buffer;
  const char* rpos;
  const char* rlast;
  
  void skip_ws();
  char readchar() throw(std::overflow_error);
  void putback();

public:
  Parser( const char* line ) : buffer(line), rpos(line)
  { rlast = rpos + strlen(line); }
  
  /** Return the string of the remainder part of the line
  */
  void ParseGetLine( std::string& ident );

  /** Parse C/C++ identifier and return the last unrecognized char 
      which is putted back into the stream
      [a-zA-Z_]*[0-9]
  */
  int ParseIdentifier( std::string& ident );

  /* arithmetic grammar ... */
  /**/
  SgUnsignedLongVal* ParseExprIntegralNumber( );

  /**/
  SgVarRefExp* ParseExprIdentifier( 
      Sg_File_Info* fileInfo, 
      SgScopeStatement* scope 
  );

  /**/
  SgExpression* ParseExprConstant( 
      Sg_File_Info* fileInfo, 
      SgScopeStatement* scope 
  );

  /**/
  SgExpression* ParsePrimaryExpression( 
      Sg_File_Info* fileInfo, 
      SgScopeStatement* scope 
  );
  
  /**/
  SgExpression* ParseUnaryExpression( 
      Sg_File_Info* fileInfo, 
      SgScopeStatement* scope 
  );

  /**/
  SgExpression* ParseCastExpression( 
      Sg_File_Info* fileInfo, 
      SgScopeStatement* scope 
  );

  /**/
  SgExpression* ParseMultiplicativeExpression( 
      Sg_File_Info* fileInfo, 
      SgScopeStatement* scope 
  );

  /**/
  SgExpression* ParseAdditiveExpression( 
      Sg_File_Info* fileInfo, 
      SgScopeStatement* scope 
  );

  SgExpression* ParseExpression( 
      Sg_File_Info* fileInfo, 
      SgScopeStatement* scope 
  );

  KaapiStorage_t ParseStorage( 
      Sg_File_Info*       fileInfo, 
      SgScopeStatement*   scope 
  );


  KaapiAccessMode_t ParseAccessMode( 
      Sg_File_Info* fileInfo
  );

  KaapiReduceOperator_t* ParseReduceOperator( 
      Sg_File_Info* fileInfo
  );

  KaapiReduceOperator_t* ParseReductionDeclaration( 
      Sg_File_Info* fileInfo, 
      SgScopeStatement* scope 
  );

  void ParseDimension( 
      Sg_File_Info* fileInfo, 
      KaapiParamAttribute* kpa, 
      SgScopeStatement* scope 
  );

  void ParseNDimensions( 
      Sg_File_Info* fileInfo, 
      KaapiParamAttribute* kpa, 
      SgScopeStatement* scope 
  );

  void ParseRangeDeclaration( 
      Sg_File_Info*        fileInfo, 
      KaapiParamAttribute* kpa, 
      SgScopeStatement*    scope 
  );

  void ParseComplexView( 
      Sg_File_Info*        fileInfo, 
      KaapiParamAttribute* kpa, 
      SgScopeStatement*    scope 
  );

  void ParseListParamDecl( 
      Sg_File_Info*       fileInfo, 
      KaapiTaskAttribute* kta,
      KaapiAccessMode_t   mode,
      SgScopeStatement*   scope 
  );

  void ParseListParam( 
      Sg_File_Info*       fileInfo, 
      KaapiTaskAttribute* kta,
      SgScopeStatement*   scope 
  );

  void DoKaapiPragmaTask( SgPragmaDeclaration* sgp);
  void DoKaapiPragmaSignature( SgPragmaDeclaration* sgp);
  void DoKaapiPragmaData( SgNode* node );
  void DoKaapiPragmaLoop( SgPragmaDeclaration* sgp );
  void DoKaapiTaskDeclaration( SgFunctionDeclaration* functionDeclaration );
  void DoKaapiPragmaNoTask( SgPragmaDeclaration* sgp );
  void DoKaapiPragmaBarrier( SgPragmaDeclaration* sgp );
  void DoKaapiPragmaWaiton( SgPragmaDeclaration* sgp );
  void DoKaapiPragmaParallelRegion( SgPragmaDeclaration* sgp );
  void DoKaapiPragmaInit( SgPragmaDeclaration* sgp, bool flag );
  void DoKaapiPragmaDeclare( SgPragmaDeclaration* sgp );
  
}; /* end parser class */


#endif // ! KAAPI_PARSER_H_INCLUDED
