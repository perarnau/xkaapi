      program main
      external fu
      external bar

      INCLUDE 'kaapif.inc'

! main()

      real*8 array(4096 * 48 * 10)
      real*8 diff
      real*8 start, stop
      integer*4 nchunks
      integer*4 chunk_size
      integer*4 i,j,ret
      byte ci
      real*4 f1
      real*8 f2

      chunk_size = 4096
      nchunks = 48 * 10

      ci=42
      f1=17.0
      f2=14.0

! init runtime
      ret = kaapif_init(1)

      start = kaapif_get_time()

      ret = kaapif_begin_parallel()

! spawn a write task per chunk
      do i = 1, nchunks

! write task (implemented by subroutine fu)
         ret = kaapif_spawn(6, fu,
! argument[0]
     &                     KAAPIF_MODE_W,
     &                     KAAPIF_TYPE_DOUBLE,
     &                     chunk_size,
     &                     array(((i - 1) * chunk_size) + 1),
! argument[1]
     &                     KAAPIF_MODE_V,
     &                     KAAPIF_TYPE_INT,
     &                     1,
     &                     chunk_size,
! argument[2]
     &                     KAAPIF_MODE_V,
     &                     KAAPIF_TYPE_CHAR,
     &                     1,
     &                     ci,
! argument[3]
     &                     KAAPIF_MODE_V,
     &                     KAAPIF_TYPE_INT,
     &                     1,
     &                     i,
! argument[4]
     &                     KAAPIF_MODE_V,
     &                     KAAPIF_TYPE_REAL,
     &                     1,
     &                     f1,
! argument[5]
     &                     KAAPIF_MODE_V,
     &                     KAAPIF_TYPE_DOUBLE,
     &                     1,
     &                     f2)

      end do


! spawn a read task per chunk
      do i = 1, nchunks
! read task (implemented by subroutine bar)
         ret = kaapif_spawn(2, bar,
! argument[0]
     &                     KAAPIF_MODE_R,
     &                     KAAPIF_TYPE_DOUBLE,
     &                     chunk_size,
     &                     array(((i - 1) * chunk_size) + 1),
! argument[1]
     &                     KAAPIF_MODE_V,
     &                     KAAPIF_TYPE_INT,
     &                     1,
     &                     chunk_size)

      end do

! synchronize and leave parallel region
      call kaapif_sched_sync()
      ret = kaapif_end_parallel(1)

      stop = kaapif_get_time()

! finalize runtime
      ret = kaapif_finalize()

! check the array
      do i = 1, nchunks
         do j = 1, chunk_size
            !diff = abs(array(j+chunk_size * (i - 1)) - (42.0 + (17.0 * i) + (14.0 * j) + cos(sqrt(42.0+17.0*i+14.0*j))))
            diff = array(j+chunk_size * (i - 1))
            diff = diff - (42.0 + (17.0 * i) + (14.0 * j))
            diff = diff - cos(sqrt(42.0+17.0*i+14.0*j))
            if (diff .gt. 0.0001) then
               write(*, *) 'invalid at ', i, ' == ', array(i)
            end if
         end do
      end do

      write(*, *) 'time: ', stop - start

      end program main
