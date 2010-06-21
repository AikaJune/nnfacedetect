#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "logpolar.h"

void logpolarXform( image_t *src,
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
            bilinearInterp( src, x, y, &interp );
            row[w] = interp;

        }

    }

}
