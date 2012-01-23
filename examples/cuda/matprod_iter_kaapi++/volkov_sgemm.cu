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

