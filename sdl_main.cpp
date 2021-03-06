#include <SDL2/SDL.h>
#ifdef main
#undef main
#endif
#include "cudaray.h"
#include <vector>

#ifdef HAVE_OPENMP
#include <omp.h>
#endif

static float rnd()
{
    static long a = 3;
    a = (((a * 214013L + 2531011L) >> 16) & 32767);

    return (((a % 65556) + 1)) / 65556.0f;
}

struct t_light_aux
{
    float r;
    float xangle;
    float yangle;
    float speed;
};

double time_get()
{
    uint64_t time = SDL_GetPerformanceCounter();
    uint64_t frequency = SDL_GetPerformanceFrequency();

    return ((double)time) / ((double)frequency);
}

int main( int argc, char * argv[] )
{
    int width = 800;
    int height = 800;
    int sphere_columns = 4;
    int sphere_rows = 4;
    int block_width = 32;
    int block_height = 32;
    bool cpu_mode = false;
    bool benchmark_mode = false;

    for( int i = 1; i < argc; ++i )
    {
        char * arg = argv[i];
        if( !strcmp( arg, "--cpu" ) )
            cpu_mode = true;
        else if( !strcmp( arg, "--width" ) )
        {
            i++;
            width = atoi( argv[i] );
        }
        else if( !strcmp( arg, "--height" ) )
        {
            i++;
            height = atoi( argv[i] );
        }
        else if( !strcmp( arg, "--rows" ) )
        {
            i++;
            sphere_rows = atoi( argv[i] );
        }
        else if( !strcmp( arg, "--columns" ) )
        {
            i++;
            sphere_columns = atoi( argv[i] );
        }
        else if( !strcmp( arg, "--benchmark" ) )
        {
            benchmark_mode = true;
            printf( "benchmark mode\n" );
        }
        else if( !strcmp( arg, "--block-width" ) )
        {
            i++;
            block_width = atoi( argv[i] );
        }
        else if( !strcmp( arg, "--block-height" ) )
        {
            i++;
            block_height = atoi( argv[i] );
        }
    }

    if( cpu_mode )
    {
        #ifdef HAVE_OPENMP
        omp_set_dynamic(0);
        printf( "running with OpenMP (use OMP_NUM_THREADS to change number of threads)\n" );
        #else
        printf( "running on the CPU\n" );
        #endif
    }
    else
        printf( "running on the GPU\n" );

    SDL_Init( SDL_INIT_VIDEO );

    SDL_Window * window;
    SDL_Renderer * renderer;
    SDL_CreateWindowAndRenderer( width, height, SDL_WINDOW_SHOWN, &window, &renderer );
    SDL_Texture * texture = SDL_CreateTexture( renderer, SDL_PIXELFORMAT_ARGB8888, SDL_TEXTUREACCESS_STREAMING, width, height );

    SDL_RenderSetLogicalSize( renderer, width, height );

    std :: vector< t_sphere > spheres;

    const int sphere_vertical_stride = height / sphere_rows;
    const int sphere_horizontal_stride = width / sphere_columns;

    const int n_spheres = sphere_rows * sphere_columns;

    for( int i = 0; i < n_spheres; ++i )
    {
        t_sphere sphere;

        int y = i / sphere_columns;
        int x = i - y * sphere_columns;

        vec3_set( sphere.position, sphere_horizontal_stride * (x + 0.5), sphere_vertical_stride * (y + 0.5), 0.0f );
        vec3_set( sphere.color, 0.50 + rnd() * 0.50, 0.50 + rnd() * 0.50, 0.50 + rnd() * 0.50 );

        sphere.radius = sphere_horizontal_stride / 3.0f + (rnd() - 0.5f) * 2.0f * sphere_horizontal_stride / 8.0f;
        spheres.push_back( sphere );
    }

    const size_t n_lights = 1;
    t_light * lights = (t_light *)calloc( n_lights, sizeof( t_light ) );
    t_light_aux * lights_aux = ( t_light_aux * )calloc( n_lights, sizeof( t_light_aux ) );
    for( int i = 1; i < n_lights; ++i )
    {
        t_light * light = lights + i;
        t_light_aux * aux = lights_aux + i;

        light->intensity = 0.25f + rnd();
        if( light->intensity > 1.0f )
            light->intensity = 1.0f;

        vec3_set( light->color, rnd() + 0.25f, rnd() + 0.25f, rnd() + 0.25f );
        vec3_clamp( light->color, 0.0f, 1.0f );

        aux->r = 50.0f + rnd() * 200.0f;
        aux->xangle = rnd() * 3.14;
        aux->yangle = rnd() * 3.14;
        aux->speed = rnd() * 4.0f;
    }

    lights[0].intensity = 1.0f;
    vec3_set( lights[0].color, 1.0f, 1.0f, 1.0f );
    vec3_set( lights[0].position, width / 2.0f, height / 2.0f, 500.0f );

    uint32_t * img = (uint32_t *)malloc( width * height * sizeof( uint32_t ) );
    memset( img, 0, sizeof( img ) );

    float t = 0.0f;
    float speed = 0.3f;
    int acceleration = 0;;

    double timestamp = time_get();
    int frame_counter = 0;

    double average = 0.0f;
    bool running = true;
    uint32_t average_count = 0;

    while( running )
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
                            running = false;
                            break;
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

            if( !running )
                break;
        }
        if( !running )
            break;

        vec3_set( lights[0].position, width / 2.0f, height / 2.0f, 300.0f );
        t_vec3 shift;
        vec3_set( shift, (150.0f) * cosf(t), 0.0f, (150.0f) * sinf(t) );
        vec3_add( lights[0].position, shift );

        for( int i = 1; i < n_lights; ++i )
        {
            t_light * light = lights + i;
            t_light_aux * aux = lights_aux + i;

            t_vec3 shift;
            vec3_set( shift, aux->r * cosf( aux->xangle ) * sin( aux->yangle ), aux->r * sin( aux->xangle ) * cos( aux->yangle ), aux->r * sinf( aux->yangle ) );

            vec3_set( light->position, width / 2.0f, height / 2.0f, 0.0f );
            vec3_add( light->position, shift );

            aux->xangle += speed * aux->speed;
            aux->yangle += speed * aux->speed;
        }

        if( cpu_mode )
            cuda_main_cpu( width, height, (uint32_t *)img, &spheres.front(), spheres.size(), lights, n_lights, 0, 0 );
        else
            cuda_main( width, height, (uint32_t *)img, &spheres.front(), spheres.size(), lights, n_lights, block_width, block_height );

        SDL_UpdateTexture( texture, NULL, img, width * sizeof( uint32_t ) );
        SDL_RenderClear( renderer );
        SDL_RenderCopy( renderer, texture, NULL, NULL );
        SDL_RenderPresent( renderer );
        SDL_Delay( 2 );

        t += speed;
        speed += acceleration * 0.01f;

        frame_counter++;
        if( frame_counter == 10 )
        {
            frame_counter = 0;
            double now = time_get();
            double time_per_frame = (now - timestamp) / 10.0;
            timestamp = now;

            printf( "time per frame: %fms\n", time_per_frame * 1000.0 );

            if( average_count == 1 )
                average = time_per_frame;
            else if( average_count > 1 )
                average = (average + time_per_frame) / 2.0;

            if( average_count == 5 && benchmark_mode )
                break;

            average_count++;
        }
    }

    printf( "average time per frame: %fms\n", average * 1000.0 );

    SDL_DestroyWindow( window );
    SDL_Quit();

    return 0;

}
