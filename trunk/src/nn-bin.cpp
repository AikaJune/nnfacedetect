#include <nn.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double logistic_func( double in )
{
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

#define NINPUT 2
#define NOUTPUT 1

typedef struct
{
    double input[NINPUT];
    double output[NOUTPUT];
} sample_t;

void eval_sample( neural_network_t *nn,
        sample_t sample )

{
    for ( int i = 0; i < NINPUT; i++ )
    {
        nn->il[i].v = sample.input[i];
    }
    nn_eval( nn );
    dbg_print_nn( nn, " eval " );
}

void train_sample( neural_network_t *nn,
        sample_t sample,
        int niter )
{
    for ( int i = 0; i < NINPUT; i++ )
    {
        nn->il[i].v = sample.input[i];
    }
    for ( int i = 0; i < niter; i++ )
    {
        nn_eval( nn );
        nn_backpropagate( nn, sample.output,
                NOUTPUT );
    }
}

void train_samples( neural_network_t *nn,
        sample_t* samples,
        int nsamples,
        int nglobal_iter,
        int nlocal_iter )
{
    for ( int i = 0; i < nglobal_iter; i++ )
    {
        for ( int j = 0; j < nsamples; j++ )
        {
            train_sample( nn, samples[j], nlocal_iter );
        }
    }
}


int main()
{
    // set up sample
    sample_t s[4];// s1, s2, s3, s4;
    s[0].input = {0, 0};
    s[0].output = { 0 };
    s[1].input = {0, 1};
    s[1].output = { 1 };
    s[2].input = {1, 0};
    s[2].output = { 1 };
    s[3].input = {1, 1};
    s[3].output = { 0 };

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


    // train samples
    train_samples( &nn, s, 4, 1000000, 1 );

    eval_sample( &nn, s[0] );
    eval_sample( &nn, s[1] );
    eval_sample( &nn, s[2] );
    eval_sample( &nn, s[3] );

    nn_free( &nn );
    return 0;
}