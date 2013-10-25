#include <stdio.h>
#include <stdint.h>

__global__ 
void cuda_run(uint32_t * img, int width) 
{
	int x = blockIdx.x;
	int y = blockIdx.y;	
	int r = (int)((255.0/200.0)*x);
	img[ y * width + x ] = 0xff000000 | (r << 16);
}
 
void cuda_main( int width, int height, uint32_t * img )
{
    uint32_t * cuda_img;
    const int size = width * height * sizeof( uint32_t );
    
    cudaMalloc( (void **)&cuda_img, size );
    cudaMemset( cuda_img, 0, size );
    
    dim3 dimBlock( 1, 1 );
	dim3 dimGrid( width, height );
	
	cuda_run<<<dimGrid, dimBlock>>>( cuda_img, width );
	
	cudaMemcpy( img, cuda_img, size, cudaMemcpyDeviceToHost ); 
	cudaFree( cuda_img );
}
