 Commande pour compiler:
 g++ -c -o transform.o  -I ../ transform.cpp
 g++  -o transform  transform.o random.o  ../libxkaapi.a  -lpthread -lnuma
 CKAAPI_CPUSET=5120  numactl --interleave=5,6 ./transform 20000 2 100000

> kaapi_adapt_processor.c //modifie
>  timing.h //modifie
> CMakeCache.txt CMakeFiles cmake_install.cmake //A supprimer avant un nouveau cmake
