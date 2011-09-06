subroutine apply(x)
!
  real x
!
  x = x + 1
end subroutine apply


subroutine init(array, n)
!
  real array
  integer n
  dimension array(n)
!
  do i = 1, n
     array(i) = 1
  end do
end subroutine init


subroutine foreach(array, n)
!
  real array
  integer n
  dimension array(n)
!
!$OMP PARALLEL
!$OMP DO
  do i = 1, n
     call apply(array(i))
  end do
!$OMP END DO
!$OMP END PARALLEL
end subroutine foreach


subroutine print(array, n)
!
  real array
  integer n
  dimension array(n)
!
  do i = 1, n
     write( *, * ) array(i)
  end do
end subroutine print

program main
  real array(4096 * 500)
  integer n
  n = 4096 * 500
  call init(array, n)
  do iter = 1, 100
     call foreach(array, n)
  end do
!  call print(array, n)
end
