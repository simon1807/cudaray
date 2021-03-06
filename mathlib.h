#ifndef MATHLIB_H
#define MATHLIB_H

#ifdef __CUDACC__
#define PREFIX __device__
#else
#define PREFIX
#endif

#include <math.h>
#include <string.h>

typedef float t_vec3[3];

struct t_sphere
{
    t_vec3 position;
    float radius;
    t_vec3 color;
};

PREFIX static inline void vec3_zero( t_vec3 out )
{
    memset( out, 0, sizeof( t_vec3 ) );
}

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

PREFIX static inline void vec3_clamp( t_vec3 self, float min, float max )
{
    self[0] = (self[0] < min) ? (min) : (self[0]);
    self[0] = (self[0] > max) ? (max) : (self[0]);

    self[1] = (self[1] < min) ? (min) : (self[1]);
    self[1] = (self[1] > max) ? (max) : (self[1]);

    self[2] = (self[2] < min) ? (min) : (self[2]);
    self[2] = (self[2] > max) ? (max) : (self[2]);
}

PREFIX static inline void vec3_scalar_mul( t_vec3 self, t_vec3 in )
{
    self[0] *= in[0];
    self[1] *= in[1];
    self[2] *= in[2];
}

PREFIX static inline void vec3_direction( t_vec3 self, t_vec3 origin, t_vec3 destination )
{
    vec3_dup( self, destination );
    vec3_sub( self, origin );
    vec3_normalize( self );
}

#endif
