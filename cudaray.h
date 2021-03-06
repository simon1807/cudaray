#ifndef CUDARAY_H
#define CUDARAY_H

#include "stdint.h"
#include "mathlib.h"

struct t_light
{
    t_vec3 position;
    t_vec3 color;
    float intensity;
};

void cuda_main( int width, int height, uint32_t * img, t_sphere * sphere_array, int sphere_count, t_light * light_array, int light_count, int block_width, int block_height );
void cuda_main_cpu( int width, int height, uint32_t * img, t_sphere * sphere_array, int sphere_count, t_light * light_array, int light_count, int block_width, int block_height );

#endif
