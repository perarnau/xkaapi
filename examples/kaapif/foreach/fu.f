
      subroutine fu(i, j, tid, array)
      !$ use omp_lib
      ! process [i, j]

      integer*4 i
      integer*4 j
      integer*4 tid
      real*8 array(*)

      ! unused, avoid warning
      ! write(*, *) tid
      tid = tid

      do k = 1, j
         array(k) = sqrt(sin(array(k)) * cos(array(k)))
         !array(k) = array(k) + 1
      end do

      return
      end
