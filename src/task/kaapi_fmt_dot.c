
#include "kaapi_impl.h"

void kaapi_fmt_set_dot_name( kaapi_format_t* fmt, const char* name )
{
    fmt->name_dot = name;
}

void kaapi_fmt_set_dot_color( kaapi_format_t* fmt, const char* color )
{
    fmt->color_dot = color;
}

