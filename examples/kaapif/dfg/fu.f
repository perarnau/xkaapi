      subroutine fu(array, size)

      ! process array[0, size[
      ! process array[1, size]

      real*8 array(*)
      integer*4 size
      integer*4 i

      ! integer*4 tid
      ! tid = kaapif_get_thread_num()
      ! write(*, *) tid, 'FU', size

      do i = 1, size
         array(i) = 42
      end do

      return
      end
