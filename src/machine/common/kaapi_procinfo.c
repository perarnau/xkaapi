/*
 ** kaapi_procinfo.h
 ** 
 ** Created on Jun 23 2010
 ** Copyright 2009 INRIA.
 **
 ** Contributors :
 **
 ** thierry.gautier@inrialpes.fr
 ** fabien.lementec@imag.fr
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
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include "kaapi_impl.h"
#include "kaapi_procinfo.h"


/** procset grammar
 proc_set   : proc_expr ( ',' proc_expr ) *
 proc_expr  : '!'* proc_list '~' proc_index
 proc_list  : proc_index ':' proc_index |
 ':' proc_index |
 ':'
 proc_index : num
 num        : [ '0' '9' ]+
 */

struct procset_parser
{
# define PROC_BIT_IS_DISABLED (1 << 0)
# define PROC_BIT_IS_USABLE (1 << 1)
  unsigned char proc_bits[KAAPI_MAX_PROCESSOR];
  unsigned int proc_binding[KAAPI_MAX_PROCESSOR];
  unsigned int proc_count;
  unsigned int max_count;
  const char* str_pos;
  int err_no;
};


typedef struct procset_parser procset_parser_t;


/**
 */
static void init_parser(
                        procset_parser_t* parser, 
                        const char* str_pos, 
                        unsigned int max_count
                        )
{
  memset(parser->proc_bits, 0, sizeof(parser->proc_bits));
  parser->proc_count = 0;
  parser->max_count = max_count;
  parser->str_pos = str_pos;
  parser->err_no = 0;
}


/**
 */
static inline void set_parser_error (procset_parser_t* parser, int err_no)
{
  parser->err_no = err_no;
}


/**
 */
static int parse_proc_index(procset_parser_t* parser, unsigned int* index)
{
  char* end_pos;
  
  *index = (unsigned int)strtoul(parser->str_pos, &end_pos, 10);
  
  if (end_pos == parser->str_pos)
  {
    set_parser_error(parser, EINVAL);
    return -1;
  }
  
  parser->str_pos = end_pos;
  
  return 0;
}


/**
 */
static inline int is_digit(const int c)
{
  return (c >= '0') && (c <= '9');
}


/**
 */
static inline int is_list_delim(const int c)
{
  return c == ':';
}


/**
 */
static inline int is_eol(const int c)
{
  /* end of list */
  return (c == ',') || (c == 0);
}


/**
 */
static inline int is_binding(const int c)
{
  return c == '~';
}


/**
 */
static int parse_proc_list(procset_parser_t* parser, unsigned int* index_low, unsigned int* index_high)
{
  /* 1 token look ahead */
  if (is_digit(*(parser->str_pos)))
  {
    if (parse_proc_index(parser, index_low))
      return -1;
  }
  else
  {
    *index_low = 0;
  }
  
  *index_high = *index_low;
  
  if (is_list_delim(*parser->str_pos))
  {
    ++parser->str_pos;
    
    /* 1 token look ahead */
    if (is_eol(*(parser->str_pos)))
      *index_high = parser->max_count - 1;
    else if (parse_proc_index(parser, index_high))
      return -1;
  }
  
  /* swap indices if needed */
  if (*index_low > *index_high)
  {
    const unsigned int tmp_index = *index_high;
    *index_high = *index_low;
    *index_low = tmp_index;
  }
  
  return 0;
}


/**
 */
static inline int parse_proc_binding(procset_parser_t* parser, unsigned int* index)
{
  /* assume beginning of binding already checked */
  /* for the moment, only one binding possible */
  
  char* end_pos;
  
  *index = (unsigned int)strtoul(parser->str_pos, &end_pos, 10);
  
  if (end_pos == parser->str_pos)
  {
    set_parser_error(parser, EINVAL);
    return -1;
  }
  
  parser->str_pos = end_pos;
  
  return 0;
}


/**
 */
static int parse_proc_expr(procset_parser_t* parser)
{
  unsigned int is_enabled = 1;
  unsigned int index_low;
  unsigned int index_high;
  unsigned int index_binding;
  
  for (; *parser->str_pos == '!'; ++parser->str_pos)
    is_enabled ^= 1;
  
  if (parse_proc_list(parser, &index_low, &index_high))
    return -1;
  
  /* an unbound proc is bound to the cpu with the same index */
  index_binding = index_low;
  if (is_binding(*parser->str_pos))
  {
    ++parser->str_pos;
    if (parse_proc_binding(parser, &index_binding))
      return -1;
  }
  
  for (; index_low <= index_high; ++index_low, ++index_binding)
  {
    if (is_enabled == 0)
      parser->proc_bits[index_low] |= PROC_BIT_IS_DISABLED;
    
    if (parser->proc_bits[index_low] & PROC_BIT_IS_DISABLED)
    {
      /* this proc is disabled. if it has been previously
       mark usable, discard it and decrement the cpu count
       */
      
      if (parser->proc_bits[index_low] & PROC_BIT_IS_USABLE)
      {
        parser->proc_bits[index_low] &= ~PROC_BIT_IS_USABLE;
        --parser->proc_count;
      }
    }
    else if (!(parser->proc_bits[index_low] & PROC_BIT_IS_USABLE))
    {
      /* proc not previously usable */
      
      parser->proc_bits[index_low] |= PROC_BIT_IS_USABLE;
      parser->proc_binding[index_low] = index_binding;
      
      if (++parser->proc_count == parser->max_count)
        break;
    }
  }
  
  return 0;
}


/**
 */
static int parse_string(procset_parser_t* parser)
{
  /* build the cpu set from string */
  
  while (1)
  {
    if (parse_proc_expr(parser))
      return -1;
    
    if (*parser->str_pos != ',')
    {
      if (!*parser->str_pos)
        break;
      
      set_parser_error(parser, EINVAL);
      
      return -1;
    }
    
    ++parser->str_pos;
  }
  
  return 0;
}


/**
 */
static void reset_procinfo_list(kaapi_procinfo_list_t* kpl)
{
  kpl->count = 0;
  kpl->head = NULL;
  kpl->tail = NULL;
}


/**
 */
kaapi_procinfo_t* kaapi_procinfo_alloc(void)
{
  kaapi_procinfo_t* const kpi = malloc(sizeof(kaapi_procinfo_t));
  if (kpi == NULL)
    return NULL;
  
  kpi->next = NULL;
  
  return kpi;
}


/**
 */
void kaapi_procinfo_free(kaapi_procinfo_t* kpi)
{
  free(kpi);
}


/**
 */
void kaapi_procinfo_list_init(kaapi_procinfo_list_t* kpl)
{
  reset_procinfo_list(kpl);
}


/**
 */
void kaapi_procinfo_list_free(kaapi_procinfo_list_t* kpl)
{
  kaapi_procinfo_t* pos = kpl->head;
  kaapi_procinfo_t* tmp;
  
  while (pos != NULL)
  {
    tmp = pos;
    pos = pos->next;
    kaapi_procinfo_free(tmp);
  }
  
  reset_procinfo_list(kpl);
}


/**
 */
void kaapi_procinfo_list_add(kaapi_procinfo_list_t* kpl, kaapi_procinfo_t* kpi)
{
  /* add tail */
  
  if (kpl->tail != NULL)
    kpl->tail->next = kpi;
  else
    kpl->head = kpi;
  kpl->tail = kpi;
  
  ++kpl->count;
}


/**
 */
int kaapi_procinfo_list_parse_string
(
 kaapi_procinfo_list_t* kpl,
 const char* procset_str,
 unsigned int proc_type,
 unsigned int max_count
 )
{
  kaapi_procinfo_t* kpi;
  procset_parser_t parser;
  unsigned int count;
  unsigned int i;
  
  /* parse string */
  init_parser(&parser, procset_str, max_count);
  if (parse_string(&parser))
    return -1;
  
  /* turn to kpl */
  count = parser.proc_count;
  for (i = 0; count && (i < KAAPI_MAX_PROCESSOR); ++i)
  {
    /* not usable */
    if (!(parser.proc_bits[i] & PROC_BIT_IS_USABLE))
      continue ;
    
    kpi = kaapi_procinfo_alloc();
    if (kpi == NULL)
      return -1;
    
    kpi->proc_index = i;
    kpi->proc_type = proc_type;
    kpi->bound_cpu = parser.proc_binding[i];
    
    kaapi_procinfo_list_add(kpl, kpi);
    
    --count;
  }
  
  return 0;
}
