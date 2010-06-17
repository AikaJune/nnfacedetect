#ifndef NN_H
#define NN_H

struct _neuron;

typedef struct _neuron neuron_t;

struct _neuron
{
	int nx;
	int nout;
	neuron_t **x; // input neurons
	neuron_t **out; // output neurons

	double *w; // weights
	double a; // a = dotprod( w, x )
	double b; // bias
	double v; // the result of act( a )

	double (*act)( double dotprod ); // activation function
	double (*actp)( double dotprod ); // activation derivative used for backprop

};

typedef struct _neural_network
{
	int ninput;
	double *input;

	int nhl;
	int nol;
	neuron_t *hl;
	neuron_t *ol;


} neural_network_t;


void nn_init( int ninput,
		int noutput,
		int nhidden,
		double (*hidden_act)(double),
		double (*hidden_actp)(double),
		double (*output_act)(double),
		double (*output_actp)(double),
	       	neural_network_t *out_nn );


#endif
