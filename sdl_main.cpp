#include <SDL2/SDL.h>
#include "cudaray.h"

int main( int argc, char * argv[] )
{
    SDL_Init( SDL_INIT_VIDEO );

    static const int width = 200;
    static const int height = 200;

    SDL_Window * window;
    SDL_Renderer * renderer;
    SDL_CreateWindowAndRenderer( width, height, SDL_WINDOW_SHOWN, &window, &renderer );
    SDL_Texture * texture = SDL_CreateTexture( renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, width, height );

    SDL_RenderSetLogicalSize( renderer, width, height );

    t_sphere sphere;
    vec3_set( sphere.position, 100.0f, 75.0, 0.0f );
    vec3_set( sphere.color, 1.0f, 1.0f, 1.0f );
    sphere.radius = 50.0f;

    const float light_distance = 30.0f;

    t_vec3 shift;
    t_light lights[4];
    vec3_set( lights[0].color, 1.0f, 0.25f, 0.25f );
    vec3_set( lights[1].color, 0.25f, 0.25f, 1.0f );
    vec3_set( lights[2].color, 0.25f, 1.0f, 0.25f );
    vec3_set( lights[3].color, 0.25f, 0.25f, 0.25f );

    int n_lights = sizeof( lights ) / sizeof( t_light );

    uint32_t img[ width ][ height ];
    memset( &img, 0, sizeof( img ) );

    float t = 0.0f;
    for( ;; )
    {
        SDL_Event event;
        while( SDL_PollEvent( &event ) )
        {
            switch( event.type )
            {
                case SDL_KEYUP:
                    if( event.key.keysym.sym == SDLK_ESCAPE )
                        return 0;
                    break;
            }
        }

        for( int i = 0; i < n_lights; ++i )
            vec3_dup( lights[i].position, sphere.position );

        vec3_set( shift, (sphere.radius + light_distance) * cosf(t), 0.0f, (sphere.radius + light_distance) * sinf(t) );
        vec3_add( lights[0].position, shift );

        vec3_set( shift, 0.0f, (sphere.radius + light_distance) * cosf(t), (sphere.radius + light_distance) * sinf(t) );
        vec3_add( lights[1].position, shift );

        vec3_set( shift, (sphere.radius + light_distance) * cosf(t * 0.75f), (sphere.radius + light_distance) * cosf(t * 0.75f), (sphere.radius + light_distance) * sinf(t * 0.75f) );
        vec3_add( lights[2].position, shift );

        vec3_set( shift, 0.0f, 0.0f, sphere.radius + 200.0f );
        vec3_add( lights[3].position, shift );

        cuda_main( width, height, (uint32_t *)img, &sphere, 1, lights, n_lights );

        SDL_UpdateTexture( texture, NULL, img, width * sizeof( uint32_t ) );
        SDL_RenderClear( renderer );
        SDL_RenderCopy( renderer, texture, NULL, NULL );
        SDL_RenderPresent( renderer );
        SDL_Delay( 16 );

        t += 0.3;
    }

    SDL_DestroyWindow( window );
    SDL_Quit();

    return 0;

}
