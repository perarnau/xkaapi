#include <cblas.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define _GNU_SOURCE
#define _XOPEN_SOURCE 600
#include "sched.h"
#include "pthread.h"

typedef struct {
	double* A ;
	double* B ;
	double* C ;
	unsigned int N ;
	pthread_barrier_t* barrier ;
	double exp_time ;
} work_t ;

void matrix_mult ( void* ptr )
{
	struct timespec m_time_s, m_time_f ;
	double exp_time ;
	work_t* work = (work_t*) ptr ;
	pthread_barrier_wait ( work->barrier ) ;
	clock_gettime ( CLOCK_REALTIME, &m_time_s ) ;
	cblas_dgemm (CblasRowMajor, CblasNoTrans, CblasNoTrans,
				 work->N, work->N, work->N,
				 1.0, work->A, work->N, work->B, work->N,
				 0.0, work->C, work->N);
	clock_gettime ( CLOCK_REALTIME, &m_time_f ) ;
	work->exp_time = (double)(m_time_f.tv_sec - m_time_s.tv_sec)
			       + (double)(m_time_f.tv_nsec - m_time_s.tv_nsec) / 1e9 ;
}

int main (int argc, char** argv)
{
	if ( argc != 3 ) {
		printf("bad args : N num_th\n");
		return 0 ;
	}
	srand(time(NULL));
	unsigned int N = atol ( argv[1] ) ;
	unsigned int num_th = atoi ( argv[2] ) ;
	unsigned int i,j ;
	work_t work[num_th+1] ;
	pthread_barrier_t barrier[2] ;
	pthread_barrier_init ( &barrier[0], NULL, 1 ) ;
	pthread_barrier_init ( &barrier[1], NULL, num_th ) ;



	printf("init\n");
	for ( i = 0 ; i <= num_th ; ++i ) {
		work[i].A = malloc ( N*N*sizeof(double) ) ;
		work[i].B = malloc ( N*N*sizeof(double) ) ;
		work[i].C = malloc ( N*N*sizeof(double) ) ;
		for( j = 0; j < N*N; ++j) {
			work[i].A[j] = (double)rand()/(double)RAND_MAX;
			work[i].B[j] = (double)rand()/(double)RAND_MAX;
			work[i].C[j] =  0 ;
		}
		work[i].barrier = &barrier[i>0] ;
		work[i].N = N ;
	}

	
	pthread_t th[num_th+1] ;

	printf("seq\n");
	pthread_create ( &th[0], NULL, (void* (*)(void*)) &matrix_mult, (void*) &work[0] ) ;
	pthread_join ( th[0], NULL ) ;

	printf("par\n");
	for ( i = 1 ; i <= num_th ; ++i )
		pthread_create ( &th[i], NULL, (void* (*)(void*)) &matrix_mult, (void*) &work[i] ) ;
	for ( i = 1 ; i <= num_th ; ++i )
		pthread_join ( th[i], NULL ) ;

	for ( i = 0 ; i <= num_th ; ++i )
		printf ( "%d -> %f (ms)\n", i, 1e3 * work[i].exp_time ) ;

	printf("clean\n");
	pthread_barrier_destroy ( &barrier[0] ) ;
	pthread_barrier_destroy ( &barrier[1] ) ;
	for ( i = 0 ; i <= num_th ; ++i ) {
		free ( work[i].A ) ;
		free ( work[i].B ) ;
		free ( work[i].C ) ;
	}

	return 0;
}

