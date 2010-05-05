C KAAPI public interface 
C (c) INRIA, projet MOAIS, 2010
C Author: T. Gautier      
C      Experimental Fortran Interface for KAAPI 
C      The proposed interface is a high level interface closed to the C++ Kaapi interface
C

C 
C      Format definition for basic fortran type
C
       INTEGER kaapi_isleader 
       INTEGER kaapi_new_signature 

C 
C      Format definition for basic fortran type
C
       INTEGER KAAPI_BYTE 
       INTEGER KAAPI_CHARACTER 
       INTEGER KAAPI_INT 
       INTEGER KAPAI_REAL
       INTEGER KAAPI_DOUBLE_PRECISION 
       INTEGER KAAPI_DOUBLE
       INTEGER KAAPI_COMPLEX 
       INTEGER KAAPI_DOUBLE_COMPLEX 

       PARAMETER (KAAPI_BYTE=0)
       PARAMETER (KAAPI_CHARACTER=1)
       PARAMETER (KAAPI_INT=2)
       PARAMETER (KAPAI_REAL=3)
       PARAMETER (KAAPI_DOUBLE_PRECISION=4)
       PARAMETER (KAAPI_DOUBLE=4)
       PARAMETER (KAAPI_COMPLEX=5)
       PARAMETER (KAAPI_DOUBLE_COMPLEX=6)
