#include <stdio.h>

/*
 * This attribute lead gcc/ld to
 * exec this function 
 * before the "main".
 */
__attribute__ ((__constructor__)) 
 void pre_func(void) {
        printf("pre_func\n");
}
