      program main
      external fu
      external bar

      include 'kaapif.inc'

! main()

      real*8 array(4096 * 48 * 10)
      real*8 diff
      real*8 start, stop
      integer*4 nchunks
      integer*4 chunk_size
      integer*4 i

      chunk_size = 4096
      nchunks = 48 * 10

! init runtime
      call kaapif_init(1)

      start = kaapif_get_time()

      call kaapif_begin_parallel()

! spawn a write task per chunk
      do i = 1, nchunks

! write task (implemented by subroutine fu)
         call kaapif_spawn(fu, 2,
! argument[0]
     &                     KAAPIF_MODE_W,
     &                     array(((i - 1) * chunk_size) + 1),
     &                     chunk_size,
     &                     KAAPIF_TYPE_DOUBLE,
! argument[1]
     &                     KAAPIF_MODE_V,
     &                     chunk_size,
     &                     1,
     &                     KAAPIF_TYPE_INT)

      end do


! spawn a read task per chunk
      do i = 1, nchunks
! read task (implemented by subroutine bar)
         call kaapif_spawn(bar, 2,
! argument[0]
     &                     KAAPIF_MODE_R,
     &                     array(((i - 1) * chunk_size) + 1),
     &                     chunk_size,
     &                     KAAPIF_TYPE_DOUBLE,
! argument[1]
     &                     KAAPIF_MODE_V,
     &                     chunk_size,
     &                     1,
     &                     KAAPIF_TYPE_INT)

      end do

! synchronize and leave parallel region
      call kaapif_sched_sync()
      call kaapif_end_parallel(1)

      stop = kaapif_get_time()

! finalize runtime
      call kaapif_finalize()

! check the array
      do i = 1, nchunks * chunk_size
         diff = abs(array(i) - (42.0 + cos(sqrt(42.0))))
         if (diff .gt. 0.0001) then
            write(*, *) 'invalid at ', i, ' == ', array(i)
         end if
      end do

      write(*, *) 'time: ', stop - start

      end program main
