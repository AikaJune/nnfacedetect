#ifndef NN_H
#define NN_H

///////////////////////////////////////////////////////////////////////////////
// NN architecture and backpropagation
///////////////////////////////////////////////////////////////////////////////

struct _neuron;

typedef struct _neuron neuron_t;

struct _neuron
{
    int nx;
    int ny;
    neuron_t **x; // input neurons
    neuron_t **y; // output neurons

    double *w; // weights
    double a; // a = dotprod( w, x )
    double b; // bias
    double v; // the result of act( a + b )
    double d; // the delta used to compute changes in weights

    double (*act)( double dotprod ); // activation function
    double (*actp)( double dotprod ); // activation derivative used for backprop

};

typedef struct _neural_network
{

    int nil;
    int nhl;
    int nol;
    neuron_t *il;
    neuron_t *hl;
    neuron_t *ol;

    double learn_rate;


} neural_network_t;


void nn_init( int ninput,
        int noutput,
        int nhidden,
        double (*hidden_act)(double),
        double (*hidden_actp)(double),
        double (*output_act)(double),
        double (*output_actp)(double),
        double learn_rate,
        neural_network_t *out_nn );

void nn_free( neural_network_t *nn );

void nn_eval( neural_network_t *nn );


void nn_backpropagate( neural_network_t *nn,
        double *desired_output,
        int nout );

///////////////////////////////////////////////////////////////////////////////
// Training related functions
///////////////////////////////////////////////////////////////////////////////

typedef struct
{
    double *input;
    double *output;
} sample_t;


void nn_init_sample( neural_network_t *nn, sample_t *s );
void nn_free_sample(  sample_t *s );

void nn_train_samples( neural_network_t *nn,
        sample_t* samples,
        int nsamples,
        int nglobal_iter,
        int nlocal_iter,
        double *tol = 0 );

void nn_eval_sample( neural_network_t *nn,
        sample_t sample );

#endif
