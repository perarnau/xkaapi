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

      call kaapif_set_grains( 32, 32)

      ! parallel loop
      start = kaapif_get_time()
      do i = 1, 100
         call kaapif_foreach(1, size, 1, fu, array)

         ! check contents. replace by .true. to enable
         if (.true.) then
            do k = 1, size
               if (array(k) .ne. (2 + i)) then
                  write(*, *) '-- INVALID --', i, k, array(k)
                  do l=1,size
                    if (array(l) .eq. (2+i)) then
                      write(*, *) '-- VALID --[', l, ']-> ', array(l)
                    else
                      write(*, *) '-- INVALID --[', 1, ']->', array(l)
                    end if
                  end do
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
