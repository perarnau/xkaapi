      subroutine fu(array, size, cinit, chunk, f1, f2)

      ! process array[0, size[
      ! process array[1, size]

      real*8 array(*)
      integer*4 size
      byte cinit
      integer*4 chunk
      real*4 f1
      real*8 f2
      integer*4 i

      ! integer*4 tid
      ! tid = kaapif_get_thread_num()
      ! write(*, *) tid, 'FU', size

      do i = 1, size
         array(i) = cinit+f1*chunk+f2*i
      end do

      return
      end
