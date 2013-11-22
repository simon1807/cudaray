#include <SDL2/SDL.h>
#ifdef main
#undef main
#endif
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
    float speed = 0.3f;
    int acceleration = 0;;

    for( ;; )
    {
        SDL_Event event;
        while( SDL_PollEvent( &event ) )
        {
            switch( event.type )
            {
                case SDL_KEYUP:
                    switch( event.key.keysym.sym )
                    {
                        case SDLK_ESCAPE:
                            return 0;
                        case SDLK_RIGHT:
                            if( !event.key.repeat )
                                acceleration -= 1;
                            break;
                        case SDLK_LEFT:
                            if( !event.key.repeat )
                                acceleration += 1;
                            break;
                    }
                    break;

                case SDL_KEYDOWN:
                    switch( event.key.keysym.sym )
                    {
                        case SDLK_RIGHT:
                            if( !event.key.repeat )
                                acceleration += 1;
                            break;
                        case SDLK_LEFT:
                            if( !event.key.repeat )
                                acceleration -= 1;
                            break;
                    }
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

        t += speed;
        speed += acceleration * 0.01f;
    }

    SDL_DestroyWindow( window );
    SDL_Quit();

    return 0;

}
