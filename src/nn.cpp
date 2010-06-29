#include "nn.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define RMAX  97451
#define SQR(x) ( (x) * (x) )

void nn_init_sample( neural_network_t *nn, sample_t *s )
{
    s->input = new double[nn->nil];
    s->output= new double[nn->nol];
}

void nn_free_sample(  sample_t *s )
{
    delete[] s->input;
    delete[] s->output;
}


void nn_eval_sample( neural_network_t *nn,
        sample_t sample )

{
    for ( int i = 0; i < nn->nil; i++ )
    {
        nn->il[i].v = sample.input[i];
    }
    nn_eval( nn );
    //dbg_print_nn( nn, " eval " );
}


void train_sample( neural_network_t *nn,
        sample_t sample,
        int niter,
        double *err2 = 0 )
{
    for ( int i = 0; i < nn->nil; i++ )
    {
        nn->il[i].v = sample.input[i];
    }
    for ( int i = 0; i < niter; i++ )
    {
        nn_eval( nn );
        nn_backpropagate( nn, sample.output,
                nn->nol );
    }

    if ( err2 )
    {
        *err2 = 0;
        for ( int i = 0; i < nn->nol; i++ )
        {
            *err2 += SQR(nn->ol[i].v - sample.output[i]);
        }
    }
}


void nn_train_samples( neural_network_t *nn,
        sample_t* samples,
        int nsamples,
        int nglobal_iter,
        int nlocal_iter,
        double *tol )
{
    double max_err2, err2, *perr2;
    perr2 = tol ? &err2 : 0;
    for ( int i = 0; i < nglobal_iter; i++ )
    {
        max_err2 = 0;
        for ( int j = 0; j < nsamples; j++ )
        {
            train_sample( nn, samples[j], nlocal_iter, perr2 );
            if ( tol && err2 > max_err2 )
            {
                max_err2 = err2;
            }
        }
        if ( tol && sqrt(max_err2) <= *tol )
        {
            printf( "tolerance reached in global iter %i\n", i );
            break;
        }
    }
}

////////////////////////////////////////////////////////
// Performs the dot product of input neurons with weigh-
// ts.
////////////////////////////////////////////////////////
inline double dotprod( double *w, neuron_t **x, int nx )
{
    double res = 0;
    for ( int i = 0; i < nx; i++ )
    {
        res += w[i] * x[i]->v;
    }
    return res;
}

void free_neuron( neuron_t *n )
{
    if ( n->nx > 0 )
    {
        delete[] n->w;
        delete[] n->x;
    }
    if ( n->ny > 0 )
    {
        delete[] n->y;
    }
}

void init_neuron( neuron_t *n,
        int nx,
        int ny	)
{
    n->nx = nx;
    n->ny = ny;
    if ( nx > 0 )
    {
        n->w = new double[nx];
        n->x = new neuron_t*[nx];

        // init weights to random numbers
        for ( int i = 0; i < nx; i++ )
        {
            n->w[i] = (double)( rand() % RMAX) / (double)(RMAX - 1 );
        }
    }
    else
    {
        n->w = 0;
        n->x = 0;
    }
    if ( ny > 0 )
    {
        n->y = new neuron_t*[ny];
    }
    else
    {
        n->y = 0;
    }
}

void nn_free( neural_network_t *nn )
{
    for ( int i = 0; i < nn->nil; i++ )
    {
        free_neuron( &nn->il[i] );
    }
    for ( int i = 0; i < nn->nhl; i++ )
    {
        free_neuron( &nn->hl[i] );
    }
    for ( int i = 0; i < nn->nol; i++ )
    {
        free_neuron( &nn->ol[i] );
    }
    delete[] nn->hl;
    delete[] nn->ol;
    delete[] nn->il;
}


void nn_init( int ninput,
        int noutput,
        int nhidden,
        double (*hidden_act)(double),
        double (*hidden_actp)(double),
        double (*output_act)(double),
        double (*output_actp)(double),
        double learnrate,
        neural_network_t *out_nn )
{
    out_nn->nil = ninput;
    out_nn->nhl = nhidden;
    out_nn->nol = noutput;

    out_nn->il = new neuron_t[ninput];
    out_nn->hl = new neuron_t[nhidden];
    out_nn->ol = new neuron_t[noutput];

    out_nn->learn_rate = learnrate;

    neuron_t *n1,*n2;


    // TODO: currently bias is always 0,
    // may want to change this in the
    // future
    for ( int i = 0; i < ninput; i++ )
    {
        n1 = &out_nn->il[i];
        init_neuron( n1, 0, nhidden ); 
        // do not specify an activation
        n1->act =  0;
        n1->actp = 0;
        n1->b = 0;
    }
    for ( int i = 0; i < nhidden; i++ )
    {
        n1 = &out_nn->hl[i];
        init_neuron( n1, ninput, noutput ); 
        n1->act =  hidden_act;
        n1->actp = hidden_actp;
        n1->b = 0;
    }
    for ( int i = 0; i < noutput; i++ )
    {
        n1 = &out_nn->ol[i];
        init_neuron( n1, nhidden, 0  ); 
        n1->act = output_act;
        n1->actp = output_actp;
        n1->b = 0;
    }


    // hook up all the neurons

    for ( int i = 0; i < ninput; i++ )
    {
        n1 = &out_nn->il[i];
        for ( int j = 0; j < nhidden; j++ )
        {
            n2 = &out_nn->hl[j];

            n1->y[j] = n2;
        }
    }	


    for ( int i = 0; i < nhidden; i++ )
    {
        n1 = &out_nn->hl[i];
        for ( int j = 0; j < ninput; j++ )
        {
            n2 = &out_nn->il[j];

            n1->x[j] = n2;
        }
        for ( int j = 0; j < noutput; j++ )
        {
            n2 = &out_nn->ol[j];

            n1->y[j] = n2;
        }
    }	

    for ( int i = 0; i < noutput; i++ )
    {
        n1 = &out_nn->ol[i];
        for ( int j = 0; j < nhidden; j++ )
        {
            n2 = &out_nn->hl[j];

            n1->x[j] = n2;
        }
    }	

}

void eval_neuron( neuron_t *n )
{
    // dot product the weights
    // with inputs
    if ( n->nx > 0 )
    {
        n->a = dotprod( n->w,
                n->x,
                n->nx );
    }
    else
    {
        n->a = 0;
    }

    // pass through activation
    // function
    if ( n->act )
    {
        n->v =  n->act(n->a + n->b);
    }
    else
    {
        n->v = 0;
    }


}


void nn_eval( neural_network_t *nn )
{
    // do nothing with input layer!

    for( int i = 0; i < nn->nhl; i++ )
    {
        eval_neuron( &nn->hl[i] );
    }

    for( int i = 0; i < nn->nol; i++ )
    {
        eval_neuron( &nn->ol[i] );
    }
}

inline double compute_output_delta( neuron_t *on, double t )
{
#ifdef DEBUG
    //return (  t - on->v ) * on->v * ( 1 - on->v );
    if ( fabs( on->v * ( 1- on->v ) - on->actp( on->a ) ) > .00000001 )
    {
        printf( "error1!\n" );
    }
#endif
    return (  t - on->v ) * on->actp( on->a  );
}

inline double compute_hidden_delta( neuron_t *hn, int hidden_index )
{
    double res;
    neuron_t *n;

    res = 0;
    for ( int i = 0; i < hn->ny; i++ )
    {
        n = hn->y[i];
        res += n->d * n->w[hidden_index];
    }


#ifdef DEBUG
    if ( fabs( hn->v * ( 1- hn->v ) - hn->actp( hn->a ) ) > .000000001 )
    {
        printf( "error2!\n" );
    }
#endif
    res *= hn->actp( hn->a );//hn->v * ( 1 - hn->v );

    return res;
}


void nn_backpropagate( neural_network_t *nn,
        double *desired_output,
        int nout )
{
    if ( nout != nn->nol )
    {
        fprintf( stderr, "err: number of desired output "
                "do not match actual number of "
                "output in the nn\n" );
        exit( -1 );
    }

    // calculate the deltas
    for( int i = 0; i < nn->nol; i++ )
    {
        nn->ol[i].d = compute_output_delta( &nn->ol[i],
                desired_output[i] );
    }
    for ( int i = 0; i < nn->nhl; i++ )
    {
        nn->hl[i].d = compute_hidden_delta( &nn->hl[i], i );
    }



    // update weights using deltas
    neuron_t *n;
    for( int i = 0; i < nn->nol; i++ )
    {
        n = &nn->ol[i];
        for ( int j = 0; j < n->nx; j++ )
        {
            //printf( "---- w=%g u=%g\n", n->w[j], nn->learn_rate * n->d * n->x[j]->v );
            n->w[j] += nn->learn_rate * n->d * n->x[j]->v;

        }
    }
    for( int i = 0; i < nn->nhl; i++ )
    {
        n = &nn->hl[i];
        for ( int j = 0; j < n->nx; j++ )
        {
            n->w[j] += nn->learn_rate * n->d * n->x[j]->v;
        }
    }
}



