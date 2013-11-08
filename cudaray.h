#ifndef CUDARAY_H
#define CUDARAY_H

#include "stdint.h"
#include "mathlib.h"

void cuda_main( int width, int height, uint32_t * img, t_sphere * sphere_array, int sphere_count );

#endif
