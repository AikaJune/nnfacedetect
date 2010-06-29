#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "logpolar.h"

void logpolar_xform( image_t *src,
        image_t *dst,
        int nWedges,
        int nRings,
        int maxDist,
        double *centerx,
        double *centery )
{
    if ( src->nchan != 1 || src->depth != 8 )
    {
        fprintf( stderr, "err: log polar x-form supports only 1channel 8bit image\n" );
        return;
    }

    assert( nWedges == dst->ncols 
            && dst->nrows == nRings );

    double wss, rss; // step sizes
    wss = ( 2 * M_PI ) / (nWedges); 
    rss = ( log( maxDist ) ) / (nRings - 1); 

    double ws, rs;
    double x, y;

    double cx, cy;

    if ( centerx && centery )
    {
        cx = *centerx;
        cy = *centery;
    }
    else
    {
        cx = src->ncols/2;
        cy = src->nrows/2;
    }

    double interp = 0;
    for ( int r = 0; r < nRings; r++ )
    {
        unsigned char *row = dst->data + r * nWedges;
        for ( int w = 0; w < nWedges; w++ )
        {
            ws = w * wss;
            rs = r * rss;
            x = exp( rs ) * cos( ws ) + cx;
            y = exp( rs ) * sin( ws ) + cy;
            image_bilinear_interp1chan( src, x, y, &interp );
            row[w] = interp;

        }

    }

}


double atan2( double y, double x )
{
    double r = atan( fabs( y / x ) );
    if ( x < 0 && y >= 0 )
        r = M_PI - r;
    else if( x < 0 && y < 0 )
        r = M_PI + r;
    else if( x > 0 && y < 0 )
        r = 2*M_PI - r;
    else if ( x == 0 && y > 0 )
        r = M_PI /2 ;
    else if ( x == 0 && y < 0 )
        r = M_PI / 2 * 3;
    else if ( x >= 0 && y ==0 )
        r = 0;
    return r;
}

void logpolar_inv_xform( image_t *src,
        image_t *dst,
        int nWedges,
        int nRings,
        int maxDist )
{
    int n = 2*maxDist + 1;
    int c = maxDist + 1;
    assert( dst->ncols == n
            && dst->ncols == n );
    
    double t, r, R, J, interp;
    for ( int x = -maxDist-1; x < maxDist; x++ )
    {
        for ( int y = -maxDist-1; y < maxDist; y++ )
        {
            t = atan2( y,x );
            r = sqrt( x*x + y*y );
            r = r < 1 ? 1 : r;
            R = log( r ) / log( maxDist ) * ( nRings - 1 );
            J = t / ( 2 * M_PI ) * (nWedges);
            image_bilinear_interp1chan( src, J, R, &interp );
            dst->data[(c+y)*n + c+x] = interp;
        }
    }
}
