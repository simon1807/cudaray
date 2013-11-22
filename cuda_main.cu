#include <stdio.h>
#include <stdint.h>

#include "cudaray.h"
#include "mathlib.h"

#ifndef __CUDACC__ /* Ghetto CUDA. */
    #define __device__
    #define __global__

    #include <stdlib.h>
    #include <string.h>

    static void cudaMalloc( void * p, size_t size )
    {
        void ** out = (void **)p;
        *out = malloc( size );
    }

    static void cudaMemset( void * ptr, int x, size_t size )
    {
        memset( ptr, x, size );
    }

    static void cudaMemcpy( void * out, const void * in, size_t size, int direction )
    {
        memcpy( out, in, size );
    }

    static void cudaFree( void * ptr )
    {
        free( ptr );
    }

    #define cudaMemcpyHostToDevice 0
    #define cudaMemcpyDeviceToHost 0

    static struct
    {
        int x;
        int y;
    } blockIdx;
#endif

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
static int sphere_intersect( const t_ray ray, const t_sphere * sphere, t_vec3 out, float * out_distance )
{
    float r = sphere->radius;

    t_vec3 d;
    vec3_dup( d, ray.start );
    vec3_sub( d, sphere->position );

    float a = vec3_dot( ray.direction, ray.direction );
    float b = 2 * vec3_dot( d, ray.direction );
    float c = vec3_dot( d, d ) - pow2(r);
    float delta = pow2(b) - 4 * a * c;

    if( delta < 0.0f )
        return 0;

    float t;

    if( b > 0 )
        t = (-b + sqrtf( delta )) / (2 * a);
    else
        t = (-b - sqrtf( delta )) / (2 * a);

    vec3_dup( out, ray.direction );
    vec3_scale( out, t );
    vec3_add( out, ray.start );
    *out_distance = vec3_dist( ray.start, out );

    return 1;
}

__global__ 
void cuda_run( uint32_t * img, int width, t_sphere * sphere_array, int sphere_count, t_light * light_array, int light_count )
{
	int x = blockIdx.x;
	int y = blockIdx.y;

    t_ray ray;
    vec3_set( ray.start, x, y, 1000.0f );
    vec3_set( ray.direction, 0.0f, 0.0f, 1.0f );

    t_sphere * best_sphere = NULL;
    float best_distance = 0.0f;
    t_vec3 best_intersect_point;

    for( int i = 0; i < sphere_count; ++i )
    {
        t_sphere * sphere = &sphere_array[i];

        t_vec3 intersect_point;
        float distance;
        if( !sphere_intersect( ray, sphere, intersect_point, &distance ) )
            continue;

        if( best_sphere == NULL || best_distance > distance )
        {
            best_sphere = sphere;
            best_distance = distance;
            vec3_dup( best_intersect_point, intersect_point );
        }
    }

    if( best_sphere == NULL )
        return;

    t_vec3 fragment_color;
    vec3_zero( fragment_color );

    for( int n_light = 0; n_light < light_count; ++n_light )
    {
        t_light * light = &light_array[n_light];

        t_vec3 light_vector;
        vec3_direction( light_vector, best_intersect_point, light->position );

        t_vec3 normal;
        vec3_direction( normal, best_sphere->position, best_intersect_point );

        float intensity = vec3_dot( normal, light_vector );
        if( intensity < 0 )
            continue;

        intensity *= light->intensity;

        vec3_dup( ray.start, best_intersect_point );
        vec3_direction( ray.direction, best_intersect_point, light->position );

        bool unobstructed = true;
        for( int j = 0; j < sphere_count; ++j )
        {
            t_sphere * sphere = &sphere_array[j];
            if( sphere == best_sphere )
                continue;

            float distance;
            t_vec3 point;
            if( sphere_intersect( ray, sphere, point, &distance ) )
            {
                unobstructed = false;
                break;
            }
        }

        if( !unobstructed )
            continue;

        t_vec3 light_color;
        vec3_dup( light_color, light->color );
        vec3_scale( light_color, intensity );

        t_vec3 color;
        vec3_dup( color, best_sphere->color );
        vec3_scalar_mul( color, light_color );
        vec3_clamp( color, 0.0f, 1.0f );

        vec3_add( fragment_color, color );
    }

    vec3_clamp( fragment_color, 0.0f, 1.0f );
    int r = 255 * fragment_color[0];
    int g = 255 * fragment_color[1];
    int b = 255 * fragment_color[2];
    img[ y * width + x ] += 0xff000000 | (r << 16) | (g << 8) | (b);
}
 
void cuda_main( int width, int height, uint32_t * img, t_sphere * sphere_array, int sphere_count, t_light * light_array, int light_count )
{
    uint32_t * cuda_img;
    t_sphere * cuda_sphere_array;
    t_light * cuda_light_array;
    const int size = width * height * sizeof( uint32_t );
    
    cudaMalloc( &cuda_img, size );
    cudaMalloc( &cuda_sphere_array, sphere_count * sizeof( t_sphere ) );
    cudaMalloc( &cuda_light_array, light_count * sizeof( t_light ) );
    cudaMemset( cuda_img, 0, size );
    cudaMemcpy( cuda_sphere_array, sphere_array, sphere_count * sizeof( t_sphere ), cudaMemcpyHostToDevice );
    cudaMemcpy( cuda_light_array, light_array, light_count * sizeof( t_light ), cudaMemcpyHostToDevice );

    #ifdef __CUDACC__
        dim3 dimBlock( 1, 1 );
        dim3 dimGrid( width, height );
	
        cuda_run<<<dimGrid, dimBlock>>>( cuda_img, width, cuda_sphere_array, sphere_count, cuda_light_array, light_count );
    #else
        for( int y = 0; y < height; ++y )
        {
            for( int x = 0; x < width; ++x )
            {
                blockIdx.x = x;
                blockIdx.y = y;
                cuda_run( cuda_img, width, cuda_sphere_array, sphere_count, cuda_light_array, light_count );
            }
        }
    #endif
	
	cudaMemcpy( img, cuda_img, size, cudaMemcpyDeviceToHost ); 
	cudaFree( cuda_img );
    cudaFree( cuda_sphere_array );
    cudaFree( cuda_light_array );
}
