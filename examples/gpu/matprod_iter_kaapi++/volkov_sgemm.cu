#if 0 // from volov 2008

//
// (c) January 24, 2008 Vasily Volkov @ UC Berkeley
//
// Other credits:
// - Paul Leventis @ Altera Corp. for prefetching and -maxrregcount techniques
// - many thanks to Wladimir J. van der Laan @ the University of Groningen
// for his cubin disassembler (http://www.cs.rug.nl/~wladimir/decuda/)
//
// Compile with -maxrregcount 32
//

#include <assert.h>
#include <stdio.h>
#include <cuda.h>
#include <cublas.h>

void error( char *message )
{
	fprintf( stderr, "ERROR: %s\n", message );
	exit (1);
}

#define assert( condition, ... ) { if( !( condition ) ) error( __VA_ARGS__ ); }

inline void Q( cudaError_t status ) { assert( status == cudaSuccess, "CUDA fails" ); }
inline void Q( cublasStatus status ){ assert( status == CUBLAS_STATUS_SUCCESS, "CUBLAS fails" ); }

//
//	SGEMM routines
//
__device__ void saxpy( float a, float *b, float *c )
{
	c[0] += a*b[0];
	c[1] += a*b[1];
	c[2] += a*b[2];
	c[3] += a*b[3];
	c[4] += a*b[4];
	c[5] += a*b[5];
	c[6] += a*b[6];
	c[7] += a*b[7];
	c[8] += a*b[8];
	c[9] += a*b[9];
	c[10] += a*b[10];
	c[11] += a*b[11];
	c[12] += a*b[12];
	c[13] += a*b[13];
	c[14] += a*b[14];
	c[15] += a*b[15];
}

__global__ void sgemmNT( const float *A, int lda, const float *B, int ldb, float* C, int ldc, int k, float alpha, float beta )
{
	const int inx = threadIdx.x;
	const int iny = threadIdx.y;
	const int ibx = blockIdx.x * 64;
	const int iby = blockIdx.y * 16;
	const int id  = inx + iny*16;

	A += ibx + id;
	B += iby + inx + __mul24( iny, ldb );
	C += ibx + id  + __mul24( iby, ldc );
	
	float a[4] = {A[0], A[lda], A[2*lda], A[3*lda]};
	float b = B[0];
	
	const float *Blast = B + k*ldb;

	A += 4*lda;
	B += 4*ldb;
    
	__shared__ float bs[4][16];
	float c[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    
	do
	{
		float as[4] = {a[0], a[1], a[2], a[3]};
		
		bs[iny][inx] = b;
		__syncthreads();
		
		a[0] = A[0*lda];
		a[1] = A[1*lda];
		a[2] = A[2*lda];
		a[3] = A[3*lda];
		b    = B[0];
		
		saxpy( as[0], &bs[0][0], c );
		saxpy( as[1], &bs[1][0], c );
		saxpy( as[2], &bs[2][0], c );
		saxpy( as[3], &bs[3][0], c );
		
		A += 4*lda;
		B += 4*ldb;
		__syncthreads();
		
	} while( B < Blast );
	
	bs[iny][inx] = b;
	__syncthreads();
	
	saxpy( a[0], &bs[0][0], c );
	saxpy( a[1], &bs[1][0], c );
	saxpy( a[2], &bs[2][0], c );
	saxpy( a[3], &bs[3][0], c );

	for( int i = 0; i < 16; i++, C += ldc )
		C[0] = alpha*c[i] + beta*C[0];
}	

__global__ void sgemmNN( const float *A, int lda, const float *B, int ldb, float* C, int ldc, int k, float alpha, float beta )
{
	const int inx = threadIdx.x;
	const int iny = threadIdx.y;
	const int ibx = blockIdx.x * 64;
	const int iby = blockIdx.y * 16;
	const int id = inx + iny*16;
	
	A += ibx + id;
	B += inx + __mul24( iby + iny, ldb );
	C += ibx + id  + __mul24( iby, ldc );
	
	const float *Blast = B + k;
	
	float c[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    
	do
	{
		float a[4] = { A[0*lda], A[1*lda], A[2*lda], A[3*lda] };

		__shared__ float bs[16][17];
		bs[inx][iny]    = B[0*ldb];
		bs[inx][iny+4]  = B[4*ldb];
		bs[inx][iny+8]  = B[8*ldb];
		bs[inx][iny+12] = B[12*ldb];
		__syncthreads();

		A += 4*lda;
		saxpy( a[0], &bs[0][0], c );		a[0] = A[0*lda];
		saxpy( a[1], &bs[1][0], c );		a[1] = A[1*lda];
		saxpy( a[2], &bs[2][0], c );		a[2] = A[2*lda];
		saxpy( a[3], &bs[3][0], c );		a[3] = A[3*lda];	

		A += 4*lda;
		saxpy( a[0], &bs[4][0], c );		a[0] = A[0*lda];
		saxpy( a[1], &bs[5][0], c );		a[1] = A[1*lda];
		saxpy( a[2], &bs[6][0], c );		a[2] = A[2*lda];
		saxpy( a[3], &bs[7][0], c );		a[3] = A[3*lda];
		
		A += 4*lda;
		saxpy( a[0], &bs[8][0], c );		a[0] = A[0*lda];
		saxpy( a[1], &bs[9][0], c );		a[1] = A[1*lda];
		saxpy( a[2], &bs[10][0], c );		a[2] = A[2*lda];
		saxpy( a[3], &bs[11][0], c );		a[3] = A[3*lda];
		
		A += 4*lda;
		saxpy( a[0], &bs[12][0], c );
		saxpy( a[1], &bs[13][0], c );
		saxpy( a[2], &bs[14][0], c );
		saxpy( a[3], &bs[15][0], c );
		
		B += 16;
		__syncthreads();
	} while( B < Blast );
	
	for( int i = 0; i < 16; i++, C += ldc )
		C[0] = alpha*c[i] + beta*C[0]; 
}	

extern "C" void volkov_sgemm( CUstream stream, char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc )
{	
	assert( transa == 'N' || transa == 'n', "unsupported value of 'transa' in ourSgemm()" );
	assert( transb == 'N' || transb == 'n' || transb == 'T' || transb == 't' || transb == 'C' || transb == 'c',
		"invalid value of 'transb' in ourSgemm()" );

	assert( (m%64) == 0 && (n%16) == 0, "unsupported dimensions of matrix C in ourSgemm()" );
	
	dim3 grid( m/64, n/16 ), threads( 16, 4 );
	if( transb == 'N' || transb == 'n' )
	{
		assert( (k%16) == 0 && k > 0, "unsupported shared dimension in ourSgemm( 'N', 'N', ... )" );
		sgemmNN<<<grid, threads, 0, stream>>>( A, lda, B, ldb, C, ldc, k, alpha, beta );
	}
	else
	{
		assert( (k%4) == 0 && k > 4, "unsupported shared dimension in ourSgemm( 'N', 'T', ... )" );
		sgemmNT<<<grid, threads, 0, stream>>>( A, lda, B, ldb, C, ldc, k, alpha, beta );
	}
}	

#else // volov 2008

#include <cuda.h>

//
//  have to unroll some of the loops manually
//
__device__ void rank1_update( float a, const float *b, float *c )
{
	c[0] += a*b[0];
	c[1] += a*b[1];
	c[2] += a*b[2];
	c[3] += a*b[3];
	c[4] += a*b[4];
	c[5] += a*b[5];
	c[6] += a*b[6];
	c[7] += a*b[7];
	c[8] += a*b[8];
	c[9] += a*b[9];
	c[10] += a*b[10];
	c[11] += a*b[11];
	c[12] += a*b[12];
	c[13] += a*b[13];
	c[14] += a*b[14];
	c[15] += a*b[15];
}

__device__ void rankk_update( int k, const float *A, int lda, const float *b, int ldb, float *c )
{
    if( k <= 0 ) return;

    int i = 0;
    rank1_update( A[0], &b[i*ldb], c ); if( ++i >= k ) return; A += lda;
    rank1_update( A[0], &b[i*ldb], c ); if( ++i >= k ) return; A += lda;
    rank1_update( A[0], &b[i*ldb], c ); if( ++i >= k ) return; A += lda;
    rank1_update( A[0], &b[i*ldb], c ); if( ++i >= k ) return; A += lda;
    
    rank1_update( A[0], &b[i*ldb], c ); if( ++i >= k ) return; A += lda;
    rank1_update( A[0], &b[i*ldb], c ); if( ++i >= k ) return; A += lda;
    rank1_update( A[0], &b[i*ldb], c ); if( ++i >= k ) return; A += lda;
    rank1_update( A[0], &b[i*ldb], c ); if( ++i >= k ) return; A += lda;
    
    rank1_update( A[0], &b[i*ldb], c ); if( ++i >= k ) return; A += lda;
    rank1_update( A[0], &b[i*ldb], c ); if( ++i >= k ) return; A += lda;
    rank1_update( A[0], &b[i*ldb], c ); if( ++i >= k ) return; A += lda;
    rank1_update( A[0], &b[i*ldb], c ); if( ++i >= k ) return; A += lda;
    
    rank1_update( A[0], &b[i*ldb], c ); if( ++i >= k ) return; A += lda;
    rank1_update( A[0], &b[i*ldb], c ); if( ++i >= k ) return; A += lda;
    rank1_update( A[0], &b[i*ldb], c );
}

__device__ void store_block( int num, float alpha, float *c, float beta, float *C, int ldc )
{
    if( num <= 0 ) return;

    if( beta == 0 )
    {
        //
        //  for the case when C is initialized with inf or NaN
        //
        int i = 0; 
        C[0] = alpha*c[i++]; if( i >= num ) return; C += ldc;  
        C[0] = alpha*c[i++]; if( i >= num ) return; C += ldc;  
        C[0] = alpha*c[i++]; if( i >= num ) return; C += ldc;  
        C[0] = alpha*c[i++]; if( i >= num ) return; C += ldc;  
        
        C[0] = alpha*c[i++]; if( i >= num ) return; C += ldc;  
        C[0] = alpha*c[i++]; if( i >= num ) return; C += ldc;  
        C[0] = alpha*c[i++]; if( i >= num ) return; C += ldc;  
        C[0] = alpha*c[i++]; if( i >= num ) return; C += ldc;  

        C[0] = alpha*c[i++]; if( i >= num ) return; C += ldc;  
        C[0] = alpha*c[i++]; if( i >= num ) return; C += ldc;  
        C[0] = alpha*c[i++]; if( i >= num ) return; C += ldc;  
        C[0] = alpha*c[i++]; if( i >= num ) return; C += ldc;  

        C[0] = alpha*c[i++]; if( i >= num ) return; C += ldc;  
        C[0] = alpha*c[i++]; if( i >= num ) return; C += ldc;  
        C[0] = alpha*c[i++]; if( i >= num ) return; C += ldc;  
        C[0] = alpha*c[i++];
    }
    else
    {
        int i = 0; 
        C[0] = alpha*c[i++] + beta*C[0]; if( i >= num ) return; C += ldc;  
        C[0] = alpha*c[i++] + beta*C[0]; if( i >= num ) return; C += ldc;  
        C[0] = alpha*c[i++] + beta*C[0]; if( i >= num ) return; C += ldc;  
        C[0] = alpha*c[i++] + beta*C[0]; if( i >= num ) return; C += ldc;  
        
        C[0] = alpha*c[i++] + beta*C[0]; if( i >= num ) return; C += ldc;  
        C[0] = alpha*c[i++] + beta*C[0]; if( i >= num ) return; C += ldc;  
        C[0] = alpha*c[i++] + beta*C[0]; if( i >= num ) return; C += ldc;  
        C[0] = alpha*c[i++] + beta*C[0]; if( i >= num ) return; C += ldc;  

        C[0] = alpha*c[i++] + beta*C[0]; if( i >= num ) return; C += ldc;  
        C[0] = alpha*c[i++] + beta*C[0]; if( i >= num ) return; C += ldc;  
        C[0] = alpha*c[i++] + beta*C[0]; if( i >= num ) return; C += ldc;  
        C[0] = alpha*c[i++] + beta*C[0]; if( i >= num ) return; C += ldc;  

        C[0] = alpha*c[i++] + beta*C[0]; if( i >= num ) return; C += ldc;  
        C[0] = alpha*c[i++] + beta*C[0]; if( i >= num ) return; C += ldc;  
        C[0] = alpha*c[i++] + beta*C[0]; if( i >= num ) return; C += ldc;  
        C[0] = alpha*c[i++] + beta*C[0];
    }
}

//
//  C = alpha*A*B + beta*C
//
/*
	lmem = 0
	smem = 1168
	reg  = 30
	active threads = 512 
 */
 
__global__ void   sgemmNN_device( int m, int n, const float *A, int lda, 
const float *B, int ldb, float* C, int ldc, int k, float alpha, float beta )
{
	const int inx = threadIdx.x;
	const int iny = threadIdx.y;
	const int ibx = blockIdx.x * 64;
	const int iby = blockIdx.y * 16;
	const int row = ibx + inx + iny*16;
	
	A += row;
	B += inx + ( iby + iny ) * ldb;
	C += row  + iby * ldc;
	
	float c[16] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    
	__shared__ float b[16][17];
	for( ; k > 0; k -= 16 )
	{
#pragma unroll
		for( int i = 0; i < 16; i += 4 )
			b[inx][iny+i]  = B[i*ldb];
		__syncthreads();

        if( k < 16 )  break;

#pragma unroll
	    for( int i = 0; i < 16; i++, A += lda )
		    rank1_update( A[0], &b[i][0], c ); 
	    __syncthreads();
		
		B += 16;
	};

    rankk_update( k, A, lda, &b[0][0], 17, c );

    if( row >= m )  return;
    
    store_block( n - iby, alpha, c, beta, C, ldc);
}	


 

//
//  Matrix-matrix multiplications
//  See http://www.netlib.org/blas/sgemm.f
//
//
//  C = alpha*A*B + beta*C
//

void  volkov_sgemm(CUstream stream, float* C, const float* A, const float* B, int hA, int wA, int wB )
{
	int m = hA ;
	int n = wB ;
	dim3 grid( (m+63)/64, (n+15)/16 ), threads( 16, 4 );
	sgemmNN_device<<<grid, threads, 0, stream>>>( m, n, A, hA, B, wA, C, hA, wA, 1.0, 0.0 );	
}

#endif
