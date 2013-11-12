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
    sphere.radius = 50.0f;

    uint32_t img[ width ][ height ];
    memset( &img, 0, sizeof( img ) );

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

        cuda_main( width, height, (uint32_t *)img, &sphere, 1 );

        SDL_UpdateTexture( texture, NULL, img, width * sizeof( uint32_t ) );
        SDL_RenderClear( renderer );
        SDL_RenderCopy( renderer, texture, NULL, NULL );
        SDL_RenderPresent( renderer );
        SDL_Delay( 16 );
    }

    SDL_DestroyWindow( window );
    SDL_Quit();

    return 0;

}
