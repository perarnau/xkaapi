      subroutine fu(i, j, tid, array)

      ! process [i, j]

      integer*4 i
      integer*4 j
      integer*4 tid
      real*8 array(*)

      ! unused, avoid warning
      ! write(*, *) tid
      tid = tid

      do k = i, j
         array(k) = sqrt(sin(array(k)) * cos(array(k)))
         ! array(k) = array(k) + 1
      end do

      return
      end
