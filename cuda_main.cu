#include <stdio.h>
#include <stdint.h>

#include "cudaray.h"
#include "mathlib.h"

struct t_ray
{
    t_vec3 start;
    t_vec3 direction;
};

__device__
static inline float pow2( float x )
{
    return x * x;
}

__device__
static int sphere_intersect( t_ray ray, t_sphere * sphere, t_vec3 out, float * out_distance )
{
    float xo = ray.start[0];
    float yo = ray.start[1];
    float zo = ray.start[2];

    float x0 = xo;
    float y0 = yo;
    float z0 = zo;

    float xs = sphere->position[0];
    float ys = sphere->position[1];
    float zs = sphere->position[2];

    float rdx = ray.direction[0];
    float rdy = ray.direction[1];
    float rdz = ray.direction[2];

    float r = sphere->radius;

    float a = pow2(rdx) + pow2(rdy) + pow2(rdz);
    float b = 2 * ((xo - xs) * rdx + (yo - ys) * rdy + (zo - zs) * rdz);
    float c = pow2(xo - xs) + pow2(yo - ys) + pow2(zo - zs) - pow2(r);
    float delta = pow2(b) - 4 * a * c;

    if( delta < 0.0f )
        return 0;

    float t = (-b + sqrtf( delta )) / (2 * a);

    vec3_set( out, xo + t * rdx, yo + t * rdy, zo + t * rdz );
    *out_distance = vec3_dist( ray.start, out );

    return 1;
}

__global__ 
void cuda_run( uint32_t * img, int width, t_sphere * sphere_array, int sphere_count )
{
	int x = blockIdx.x;
	int y = blockIdx.y;

    t_ray ray;
    vec3_set( ray.start, x, y, -100.0f );
    vec3_set( ray.direction, 0.0f, 0.0f, 1.0f );

    t_sphere * best_sphere = NULL;
    float best_distance = 0.0f;
    t_vec3 intersect_point;

    for( int i = 0; i < sphere_count; ++i )
    {
        t_sphere * sphere = &sphere_array[i];

        float distance;
        if( !sphere_intersect( ray, sphere, intersect_point, &distance ) )
            continue;

        if( best_sphere == NULL || best_distance > distance )
        {
            best_sphere = sphere;
            best_distance = distance;
        }
    }

    if( best_sphere == NULL )
        return;

    t_vec3 light;
    vec3_set( light, 10.0f, 75.0f, 1.0f );
    vec3_sub( light, intersect_point );

    t_vec3 normal;
    vec3_dup( normal, intersect_point );
    vec3_sub( normal, best_sphere->position );

    vec3_normalize( light );
    vec3_normalize( normal );

    float intensity = vec3_dot( normal, light );
	if( intensity < 0 )
	    intensity = 0;

    int r = 255 * intensity;
	img[ y * width + x ] = 0xff000000 | (r << 16);
}
 
void cuda_main( int width, int height, uint32_t * img, t_sphere * sphere_array, int sphere_count )
{
    uint32_t * cuda_img;
    t_sphere * cuda_sphere_array;
    const int size = width * height * sizeof( uint32_t );
    
    cudaMalloc( &cuda_img, size );
    cudaMalloc( &cuda_sphere_array, sphere_count * sizeof( t_sphere ) );
    cudaMemset( cuda_img, 0, size );
    cudaMemcpy( cuda_sphere_array, sphere_array, sphere_count * sizeof( t_sphere ), cudaMemcpyHostToDevice );

    dim3 dimBlock( 1, 1 );
	dim3 dimGrid( width, height );
	
	cuda_run<<<dimGrid, dimBlock>>>( cuda_img, width, cuda_sphere_array, sphere_count );
	
	cudaMemcpy( img, cuda_img, size, cudaMemcpyDeviceToHost ); 
	cudaFree( cuda_img );
    cudaFree( cuda_sphere_array );
}
