/*
** xkaapi
** 
**
** Copyright 2009 INRIA.
**
** Contributors :
**
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
#include <iostream>
#include "kaapi++" // this is the new C++ interface for Kaapi
#include "kanet_network.h" // this is the new C++ interface for Kaapi
#include <string.h>


/*
*/
char* buffer_message;
uintptr_t remote_ptr = 0;
volatile int cnt_msg = 0;


static void service(int err, ka::GlobalId source, void* buffer, size_t sz_buffer )
{
  uintptr_t* msg = (uintptr_t*)buffer;
  remote_ptr = *msg;
  printf("%i:: receive mesg, remote addr = %lx\n", ka::System::local_gid, remote_ptr);
  fflush(stdout);
  ++cnt_msg;
}

static void service_ack(int err, ka::GlobalId source, void* buffer, size_t sz_buffer )
{
  printf("%i:: receive ack mesg\n", ka::System::local_gid);
  fflush(stdout);
  ++cnt_msg;
}

enum {
  service_id     = 128,
  service_ack_id = 129
};



/* main entry point : Kaapi initialization
*/
int main(int argc, char** argv)
{
  try {
    ka::Network::register_service( service_id,       &service );
    ka::Network::register_service( service_ack_id,   &service_ack );

    /* Join the initial group of computation : it is defining
       when launching the program by a1run.
    */
    ka::System::join_community( argc, argv );
        
    buffer_message = new (ka::Network::object.allocate(256)) char[256];
    
    /*
    */
    if (ka::Network::object.size() ==2)
    {
      if (ka::System::local_gid ==0)
      { /* send to 1 my buffer address */
        ka::OutChannel* channel = ka::Network::object.get_default_local_route( 1 );
        kaapi_assert(channel !=0);
        std::cout << "Send message handler:" << (void*)service << ", data:" << (void*)buffer_message << std::endl;
        channel->insert_am( service_id, &buffer_message, sizeof( void* ) );
        channel->sync();
        
        for (int i=0; i<10; ++i)
        {
          while (cnt_msg == i) 
          {
            ka::Network::object.poll();
            sleep(1);
          }

          std::cout << "Recv message:" << buffer_message << std::endl;
        }
      } 
      else
      {
        char msg[128];

        while (remote_ptr ==0) 
          ka::Network::object.poll();
        std::cout << "Recv remote address:" << (void*)remote_ptr << std::endl;
        
        /* remote write on node 0 */
        ka::OutChannel* channel = ka::Network::object.get_default_local_route( 0 );
        kaapi_assert(channel !=0);
        for (int i=0; i<10; ++i)
        {
          sprintf(msg, " Ceci est le message numero %i de %i", i, ka::System::local_gid);
          channel->insert_rwdma(remote_ptr, msg, strlen(msg)+1);
          channel->insert_am( service_ack_id, 0, 0 );
          channel->sync();
          sleep(1);
        }
      }      
    }
      
    ka::Network::object.dump_info();

    /*
    */
    ka::Network::object.terminate();

    /* */
    ka::System::terminate();
  }
  catch (const std::exception& E) {
    ka::logfile() << "Catch : " << E.what() << std::endl;
  }
  catch (...) {
    ka::logfile() << "Catch unknown exception: " << std::endl;
  }
  
  return 0;
}
