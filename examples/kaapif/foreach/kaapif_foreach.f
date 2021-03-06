      program main
      external fu

      real*8 array(10000 * 48)
      integer*4 size
      real*8 start
      real*8 stop
      real*8 kaapif_get_time

      ! initialize array
      size = 10000 * 48
      do i = 1, size
         array(i) = 2
      enddo

      ! init runtime
      call kaapif_init(1)

      ! By default, the grain is set to 1,1.
      ! The user may use KAAPI_SEQ_GRAIN and KAAPI_PAR_GRAIN
      ! environment variables to fix them.
      !call kaapif_set_grains( 128, 128)

      ! parallel loop
      start = kaapif_get_time()
      do i = 1, 100
         call kaapif_foreach(1, size, 1, fu, array)

         ! check contents. replace by .true. to enable
         if (.false.) then
            do k = 1, size
               if (array(k) .ne. (2 + i)) then
                  write(*, *) '-- INVALID --', i, k, array(k)
                  call flush( 0 )
                  call sleep( 1 )
                  stop
               end if
            end do
         end if
         ! end check
         ! write(*, *) "Step ", i

      end do
      stop = kaapif_get_time()

      ! finalize runtime
      call kaapif_finalize()

      ! done
      write(*, *) (stop - start)

      end program main
