#ifndef MATHLIB_H
#define MATHLIB_H

#ifdef __CUDACC__
#define PREFIX __device__
#else
#define PREFIX
#endif

#include <math.h>

typedef float t_vec3[3];

struct t_sphere
{
    t_vec3 position;
    float radius;
    t_vec3 color;
};

PREFIX static inline void vec3_set( t_vec3 out, float x, float y, float z )
{
    out[0] = x;
    out[1] = y;
    out[2] = z;
}

PREFIX static inline void vec3_add( t_vec3 self, const t_vec3 in )
{
    self[0] += in[0];
    self[1] += in[1];
    self[2] += in[2];
}

PREFIX static inline void vec3_sub( t_vec3 self, const t_vec3 in )
{
    self[0] -= in[0];
    self[1] -= in[1];
    self[2] -= in[2];
}

PREFIX static inline float vec3_dot( const t_vec3 a, const t_vec3 b )
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

PREFIX static inline void vec3_dup( t_vec3 out, const t_vec3 in )
{
    out[0] = in[0];
    out[1] = in[1];
    out[2] = in[2];
}

PREFIX static inline float vec3_dist( const t_vec3 a, const t_vec3 b )
{
    return sqrtf( (a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]) + (a[2] - b[2]) * (a[2] - b[2]) );
}

PREFIX static inline void vec3_scale( t_vec3 self, float scale )
{
    self[0] *= scale;
    self[1] *= scale;
    self[2] *= scale;
}

PREFIX static inline float vec3_length( const t_vec3 self )
{
    return sqrtf( vec3_dot( self, self ) );
}

PREFIX static inline void vec3_normalize( t_vec3 self )
{
    float k = 1.0f / vec3_length( self );
    vec3_scale( self, k );
}

#endif
