      subroutine fu(i, j, tid, array, arraytid, scal, val)

      ! process [i, j]

      integer*4 i
      integer*4 j
      integer*4 tid
      real*8 array(*)
      integer*4 arraytid(*)
      integer*4 tid2p
      real*8 scal
      real*8 val

      ! unused, avoid warning
      ! write(*, *) tid
      tid2p = 1
      do k = 1, tid
        tid2p = tid2p * 2
      end do

      do k = i, j
         array(k) = array(k) + val * scal
         if (arraytid(k) .ne. 0) then
           write(*, *) tid, '-- alreay write ', k, ' by ', arraytid(k)
           stop
         end if

         arraytid(k) = tid2p
      end do

      return
      end
