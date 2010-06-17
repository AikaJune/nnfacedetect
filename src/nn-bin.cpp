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


int main()
{
	neural_network_t nn;
	nn_init( 2,
			1,
			3,
			&logistic_func,
			&logistic_func_deriv,
			&linear_func,
			&linear_func_deriv,
			&nn );

	nn_free( &nn );
	return 0;
}
