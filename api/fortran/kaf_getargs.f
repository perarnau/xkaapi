C =========================================================
C (c) INRIA, projet MOAIS, 2008
C Author: T. Gautier, F. Wagner
C
C =========================================================
       integer FUNCTION get_fortran_argc()
C       implicit none
       kaapi_getargc = IARGC() + 1
       end function

       SUBROUTINE get_fortran_argv(i, buffer)
       implicit none
       integer i
       character buffer*128
       call getarg(i, buffer)
C   force end of string to be 0
       buffer(127:) = ''
       end SUBROUTINE
