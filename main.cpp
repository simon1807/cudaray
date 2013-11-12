#include "cudaray.h"
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include "mathlib.h"

#pragma pack(push)
#pragma pack(1)

struct t_tga
{
	uint8_t id_length;
	uint8_t col_map_type;
	uint8_t img_type;
	uint8_t color_map[5];
	uint16_t x;
	uint16_t y;
	uint16_t width;
	uint16_t height;
	uint8_t pix_depth;
	uint8_t img_desc;
};

#pragma pack(pop)

int main()
{
    t_tga header;
	memset( &header, 0, sizeof( header ) ); // Filling header with zeros
	header.img_type=2; // Uncompressed true-color image
	header.width=200;
	header.height=200;
	header.pix_depth=32; // Bits/pixel

	t_sphere sphere;
    vec3_set( sphere.position, 100.0f, 75.0, 0.0f );
    vec3_set( sphere.color, 1.0f, 1.0f, 1.0f );
    sphere.radius = 50.0f;

    t_light light;
    vec3_set( light.position, 10.0f, 75.0f, 1.0f );
    vec3_set( light.color, 1.0f, 1.0f, 1.0f );

	uint32_t img[ 200 ][ 200 ];
	memset( &img, 0, sizeof( img ) );
	cuda_main( 200, 200, (uint32_t *)img, &sphere, 1, &light, 1 );

	FILE * fp = fopen( "obrazek.tga", "wb" );

	fwrite( &header, sizeof(header), 1, fp);
	fwrite( img, sizeof(img), 1, fp);

	fclose( fp );
	return 0;
}
