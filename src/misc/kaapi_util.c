/*
** kaapi_error.h
** xkaapi
** 
**
** Copyright 2009,2010,2011,2012 INRIA.
**
** Contributors :
**
** fabien.lementec@imag.fr
** thierry.gautier@inrialpes.fr
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
#include "kaapi_util.h"
#include "kaapi_impl.h"
#include <ctype.h>
#include <stdarg.h>

struct constant {
  const char* ident;
  int64_t     value;
};

struct _kaapi_parser {
  const char*      rbuff;
  int              count_constants;
  struct constant* listconstants;
};


static void _kaapi_eat_ws( struct _kaapi_parser* kp )
{
  while (isspace(*kp->rbuff)) 
    ++kp->rbuff;
}



/* Return:
   0: if integer is successfully parsed.
   ENOENT: if empty input string
   EINVAL: if not begins by integer
*/
static int _kaapi_parse_integer( uint64_t* retval, struct _kaapi_parser* kp, char sep )
{
  *retval = 0;
  if (*kp->rbuff == 0)
    return ENOENT;

  if (!isdigit(*kp->rbuff))
    return EINVAL;

  while (isdigit(*kp->rbuff))
  {
    *retval = 10 * *retval + (*kp->rbuff - '0');
    ++kp->rbuff;
  }
  return 0;
}

/* Return:
   0: if identifier is successfully parsed.
   ENOENT: if empty input string
   EINVAL: if not begins by integer
*/
static int _kaapi_parse_ident( uint64_t* retval, struct _kaapi_parser* kp, char sep )
{
  char name[128];
  char* wpos = name;

  if (*kp->rbuff == 0)
    return ENOENT;

  if (isdigit(*kp->rbuff))
    return EINVAL;

  while (isalpha(*kp->rbuff))
  {
    *wpos = *kp->rbuff;
    ++kp->rbuff;
    ++wpos;
    if (wpos == name + 127) break;
  }
  *wpos = 0;
  
  /* search value of the ident in the table of constants 
     linear search
  */
  for (int i=0; i<kp->count_constants; ++i)
  {
    if (strcasecmp(name, kp->listconstants[i].ident) == 0)
    {
      *retval = kp->listconstants[i].value;
      return 0;
    }
  }
  
  /* not found: return EINVAL */
  return EINVAL;
}


/*
*/
int kaapi_util_parse_list( 
  uint64_t* mask, const char* str, char sep,
  int count_constants,
  ...
)
{
  struct _kaapi_parser kp;
  uint64_t value;
  va_list va_args;
  
  kp.rbuff = str;
  kp.count_constants = count_constants;
  kp.listconstants = 0;
  *mask = 0;
  
  /* read array of constants */
  if (count_constants >0)
  {
    kp.listconstants = (struct constant*)alloca( count_constants*sizeof(struct constant) );
    va_start(va_args, count_constants);
    for (int i=0; i<count_constants; ++i)
    {
      kp.listconstants[i].ident = va_arg(va_args, const char*);
      kp.listconstants[i].value = (int)va_arg(va_args, uint64_t);
    }
    va_end(va_args);
  }
  
  while (*kp.rbuff != 0)
  {
    _kaapi_eat_ws( &kp );
    int err;
    /* lookup next token */
    if (isalpha(*kp.rbuff))
      err = _kaapi_parse_ident( &value, &kp, sep );
    else
    {
      err = _kaapi_parse_integer( &value, &kp, sep );
      if (value > 63)
        return EINVAL;
      value = (1UL << value);
    }
    if (err !=0)
      return err;
    *mask |= value;
      
    
    /* look for 'sep' */
    _kaapi_eat_ws( &kp );
    if (*kp.rbuff == 0) 
      return 0;
    if (*kp.rbuff != sep)
      return EINVAL;
    ++kp.rbuff;
  }
  
  return 0;
}
