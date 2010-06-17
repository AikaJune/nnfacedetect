#include "nn.h"


double dotprod( double *w, double *x, int nx )
{
	double res = 0;
	for ( int i = 0; i < nx; i++ )
	{
		res = w[i] * x[i];
	}
	return res;
}

void free_neuron( neuron_t *n )
{
	delete[] n->w;
	delete[] n->x;
	delete[] n->out;
}

void init_neuron( neuron_t *n,
	       	int nx,
		int nout	)
{
	n->nx = nx;
	n->nout = nout;
	n->w = new double[nx];
	n->x = new neuron_t*[nx];
	n->out = new neuron_t*[nout];
}

void nn_free( neural_network_t *nn )
{
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
	delete[] nn->input;
}


void nn_init( int ninput,
		int noutput,
		int nhidden,
		double (*hidden_act)(double),
		double (*hidden_actp)(double),
		double (*output_act)(double),
		double (*output_actp)(double),
	       	neural_network_t *out_nn )
{
	out_nn->ninput = ninput;
	out_nn->nhl = nhidden;
	out_nn->nol = noutput;

	out_nn->hl = new neuron_t[nhidden];
	out_nn->ol = new neuron_t[noutput];
	out_nn->input = new double[ninput];

	neuron_t *n1,*n2;

	
	for ( int i = 0; i < nhidden; i++ )
	{
		n1 = &out_nn->hl[i];
		init_neuron( n1, ninput, noutput ); 
		n1->act =  hidden_act;
		n1->actp = hidden_actp;
	}
	for ( int i = 0; i < noutput; i++ )
	{
		n1 = &out_nn->ol[i];
		init_neuron( n1, nhidden, 0  ); 
		n1->act = output_act;
		n1->actp = output_actp;
	}


	// hook up all the neurons

	for ( int i = 0; i < nhidden; i++ )
	{
		n1 = &out_nn->hl[i];
		for ( int j = 0; j < noutput; j++ )
		{
			n2 = &out_nn->ol[j];

			n1->out[j] = n2;
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


void nn_eval( neural_network_t *nn )
{

}


