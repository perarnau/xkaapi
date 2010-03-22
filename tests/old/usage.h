/*
 *  usage.h
 *  xkaapi
 *
 *  Created by CL and TG on 25/02/09.
 *  Copyright 2009 INRIA. All rights reserved.
 *
 */
#include <stdio.h>
#include <stdlib.h>

void usage_scope ()
{
  printf ("./test_cond2 arg\n");
  printf ("\tfor arg = 1, KAAPI_SYSTEM_SCOPE is used to create threads\n");
  printf ("\tfor arg = 2, KAAPI_PROCESS_SCOPE is used to create threads\n");
  exit (0);
}

void usage_threads()
{
  printf ("./test_create1 n\n\tn : how many threads to create\n");
  exit (0);
}