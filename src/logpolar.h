#ifndef LOGPOLAR_H
#define LOGPOLAR_H
#include <image.h>

void logpolarXform( image_t *src,
        image_t *dst,
        int nWedges,
        int nRings,
        int maxDist,
        double *centerx = 0,
        double *centery = 0 );



#endif
