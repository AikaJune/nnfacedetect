#include <nn.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <getopt.h>
#include <string.h>
#include <image.h>
#include <logpolar.h>

double logistic_func( double in )
{
    ////////////////////////////////////
    // Replace with taylor expansion for
    // increase in speed. Profiling sho-
    // ws this is a bottle-neck. Actual
    // speed up is negligable :(
    ////////////////////////////////////
   // return 1 / (2 - in + in*in/2 );
    return 1 / ( 1 + exp( -in ) ); 
}

double logistic_func_deriv( double in )
{
    return logistic_func( in ) * ( 1 - logistic_func( in ) );
}

double linear_func( double in )
{
    return in;
}

double linear_func_deriv( double in )
{
    return 1;
}

#define DEBUG
#ifdef DEBUG

void dbg_print_nn( neural_network_t *nn, char *comment=  0, bool print_hidden = false )
{

    printf( "******** printing nn *********\n" );
    if ( comment )
    {
        printf( "========== comment ===========\n" );
        printf( "%s\n", comment );
    }
    printf( "======== input layer =========\n" );
    for ( int i = 0; i < nn->nil; i++ )
    {
        printf( "il[%i]:\n", i );
        printf( "v=%g\n", nn->il[i].v );
    }
    if ( print_hidden )
    {
        printf( "======== hidden layer ========\n" );
        for ( int i = 0; i < nn->nhl; i++ )
        {
            printf( "hl[%i]:\n", i );
            printf( "w: " );
            for ( int j = 0; j < nn->hl[i].nx; j++ )
            {
                printf( "%g ", nn->hl[i].w[j] );
            }
            printf( "\n" );
            printf( "x->v: " );
            for ( int j = 0; j < nn->hl[i].nx; j++ )
            {
                printf( "%g ", nn->hl[i].x[j]->v );
            }
            printf( "\n" );
            printf( "b:%g\n", nn->hl[i].b );
            printf( "a=%g\n", nn->hl[i].a );
            printf( "v=%g\n", nn->hl[i].v );
        }
    }
    printf( "======== output layer ========\n" );
    for ( int i = 0; i < nn->nol; i++ )
    {
        printf( "ol[%i]:\n", i );
        printf( "w: " );
        for ( int j = 0; j < nn->ol[i].nx; j++ )
        {
            printf( "%g ", nn->ol[i].w[j] );
        }
        printf( "\n" );
        printf( "x->v: " );
        for ( int j = 0; j < nn->ol[i].nx; j++ )
        {
            printf( "%g ", nn->ol[i].x[j]->v );
        }
        printf( "\n" );
        printf( "b:%g\n", nn->ol[i].b );
        printf( "a=%g\n", nn->ol[i].a );
        printf( "v=%g\n", nn->ol[i].v );
    }
    printf( "****** end printing nn *******\n" );
}

#endif

#define RSEED 1000

bool load_sample( char *filename, sample_t *s, int *ninput, int *noutput )
{
    FILE *stream = 0;
    stream = fopen( filename, "r" );
    if ( !stream )
    {
        fprintf( stderr, "failed to open sample file '%s'\n", filename );
        return false;
    }

    char imagePath[512];
    if ( fgets( imagePath, 512, stream ) == NULL )
    {
        fprintf( stderr, "failed to get image path from sample file '%s'\n", filename );
        fclose( stream );
        return false;
    }
    char *pch;
    pch = strchr( imagePath, '\n' );
    if ( !pch )
    {
        fprintf( stderr, "failed to read image path supplied by sample file '%s'\n", filename );
        fclose( stream );
        return false;
    }
    *pch = '\0';

    image_t src;
    if ( !image_read_rawpbm( imagePath, &src ) )
    {
        fprintf( stderr, "failed to load image file '%s' supplied by sample file '%s'\n", imagePath, filename );
        fclose( stream );
        return false;
    }  
    int n = src.ncols * src.nrows;
    // setup sample input
    if ( ninput )
    {
        *ninput = n;
    }

    s->input = new double[ n ];
    for ( int i = 0; i < n; i++ )
    {
        s->input[i] = src.data[i];
    }

    delete[] src.data;

    // setup the output
    int _noutput;
    if ( fscanf( stream, "%d\n", &_noutput ) < 1  )
    {
        fprintf( stderr, "error parsing sample file '%s'\n", filename );
        delete[] s->input;
        fclose( stream );
        return false;
    }
    if ( noutput )
    {
        *noutput = _noutput;
    }

    s->output = new double[ _noutput ];
    double t;
    for ( int i = 0; i < _noutput; i++ )
    {
        if ( fscanf( stream, "%lf", &t ) < 1 )
        {
            fprintf( stderr, "error parsing sample file '%s'\n", filename );
            delete[] s->input;
            delete[] s->output;
            fclose( stream );
            return false;
        }
        s->output[i] = t;
    }
    
    fclose( stream );
    return true;
}

int main( int argc, char *argv[] )
{

#if 0
    sample_t smp;
    int nin, nout;
    load_sample( "test.smpl", &smp, &nin, &nout ); 
    printf( "%i %i\n", nin, nout );
    for ( int i = 0; i < nin; i++ )
    {
        printf( "in[%i]=%g\n", i, smp.input[i] ); 
    }
    for ( int i = 0; i < nout; i++ )
    {
        printf( "out[%i]=%g\n", i, smp.output[i] ); 
    }
    return 0;
#endif
    // set up sample
    sample_t s[4];// s1, s2, s3, s4;
    s[0].input = new double[2];
    s[1].input = new double[2];
    s[2].input = new double[2];
    s[3].input = new double[2];
    s[0].output = new double[1];
    s[1].output = new double[1];
    s[2].output = new double[1];
    s[3].output = new double[1];

    s[0].input[0] = 0;
    s[0].input[1] = 0;
    s[0].output[0] =  0 ;

    s[1].input[0] = 0;
    s[1].input[1] = 1;
    s[1].output[0] =  1 ;

    s[2].input[0] = 1;
    s[2].input[1] = 0;
    s[2].output[0] =  1 ;

    s[3].input[0] = 1;
    s[3].input[1] = 1;
    s[3].output[0] =  0 ;

    srand( RSEED );
    neural_network_t nn;

    //////////////////////////////////////////
    // NB: massive increase in effectiveness
    // of nn when switching from logistic 
    // activation function for output layer
    // to linear!
    //////////////////////////////////////////
    nn_init( 2,	// input
            1,                        // output
            6,                       // hidden
            &logistic_func,
            &logistic_func_deriv,
            //&logistic_func,           
            //&logistic_func_deriv,
            &linear_func,
            &linear_func_deriv,
            1.0,
            &nn );


    double tol = .0001;
    // train samples
    nn_train_samples( &nn, s, 4, 1000000, 1, &tol );

    nn_eval_sample( &nn, s[0] );
    dbg_print_nn( &nn );
    nn_eval_sample( &nn, s[1] );
    dbg_print_nn( &nn );
    nn_eval_sample( &nn, s[2] );
    dbg_print_nn( &nn );
    nn_eval_sample( &nn, s[3] );
    dbg_print_nn( &nn );


    nn_free( &nn );
    return 0;
}
