      program main

      character(40) version

      call kaapif_init(1)
      call kaapif_get_version(version)
      call kaapif_finalize()

      write(*, *) 'VERSION: ', version(1:40)

      end program main
