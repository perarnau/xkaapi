      program main
      external fu

      include 'kaapif.inc'

      REAL*8 kaapif_get_time

      real*8 array(1000 * 48)
      integer*4 arraytid(1000 * 48)
      integer*4 size
      real*8 start
      real*8 stop

      real*8 scale
      real*8 value

      scale = 0.5
      value = 2.0

      ! initialize array
      size = 1000 * 48
      do i = 1, size
         array(i) = 2.0
         arraytid(i) = 0
      enddo

      ! init runtime
      call kaapif_init(1)

      ! parallel loop
      start = kaapif_get_time()
      do i = 1, 100000
         call kaapif_foreach_with_format(1, size, 4, fu, 
! argument[0]
     &                                   KAAPIF_MODE_RW,
     &                                   array,
     &                                   size,
     &                                   KAAPIF_TYPE_DOUBLE,
! argument[1]
     &                                   KAAPIF_MODE_RW,
     &                                   arraytid,
     &                                   size,
     &                                   KAAPIF_TYPE_INT,
! argument[2]
     &                                   KAAPIF_MODE_V,
     &                                   scale,
     &                                   1,
     &                                   KAAPIF_TYPE_DOUBLE,
! argument[3]
     &                                   KAAPIF_MODE_V,
     &                                   value,
     &                                   1,
     &                                   KAAPIF_TYPE_DOUBLE)

         ! check contents. replace by .true. to enable
         if (.true.) then
            do k = 1, size
               if (array(k) .ne. (2.0 + i)) then
                  write(*, *) '-- INVALID --', i, k, array(k),
     &                  ' by ', arraytid(k)
                  call flush( 0 )
                  call sleep( 1 )
                  stop
               end if
            end do
         end if
         ! end check
      if ( mod( i, 1000) .eq. 0) then
         write(*, *) i, ' -- OK --'
      end if
      do l = 1, size
         arraytid(l) = 0
      enddo

      end do
      stop = kaapif_get_time()

      ! finalize runtime
      call kaapif_finalize()

      ! done
      write(*, *) (stop - start)

      end program main
