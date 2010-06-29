#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<ctype.h>
#include<math.h>
#include <assert.h>
#include "image.h"

void image_normalize( image_t *src, image_t *dst )
{
    assert( src->ncols == dst->ncols
            && src->nrows == dst->ncols
            && src->nchan == 1
           && dst->nchan == 1
           && src->depth == 8
           && src->depth == 8 );
    // calulate mean & std 
    unsigned long long sum2, sum;
    sum2 = 0;
    sum = 0;
    unsigned char v;
    unsigned int n = src->ncols * src->nrows;
    for ( int i = 0; i < src->ncols * src->nrows; i++ )
    {
        v = src->data[i];
        sum2 += v*v;
        sum +=  v;
    }
    double norm, mean;
    mean = sum / double(n);
    norm = sum2;
    norm -= n * mean * mean;
    norm = sqrt(norm);
    for ( int i = 0; i < src->ncols * src->nrows; i++ )
    {
        dst->data[i] = (double(src->data[i] - mean) / norm + 1 ) * 127.5;
    }
}

void image_bilinear_interp1chan( image_t *src, double x, double y, double *dst )
{
    double top, bot, left, right;
    double dlx, drx, dty, dby;
    left = floor(x);
    right = ceil(x);
    top = floor(y);
    bot = ceil(y);
    dlx = right - x;
    drx = x - left;
    dty = bot - y;
    dby = y - top;
   

    if ( x < 0 || x >= src->ncols
            || y < 0 || y >= src->nrows
            || left < 0 || left >= src->ncols
            || right < 0 || right >= src->ncols 
            || top < 0 || top >= src->nrows
            || bot < 0 || bot >= src->nrows  )
    {
        int itop, ibot, ileft, iright;
        itop = top;
        ibot = bot;
        ileft = left;
        iright = right;
        itop = itop % src->nrows;
        if ( itop < 0 )
            itop += src->nrows;
        ibot = ibot % src->nrows;
        if ( ibot < 0 )
            ibot += src->nrows;
        ileft = ileft % src->ncols;
        if ( ileft < 0 )
            ileft += src->ncols;
        iright = iright % src->ncols;
        if ( iright < 0 )
            iright += src->ncols;
        top = itop;
        bot = ibot;
        left = ileft;
        right = iright;
    }

    unsigned char *top_row, *bot_row;
    int byte_step = src->depth >> 3;
    top_row = src->data + (int)top * byte_step * src->ncols;
    bot_row = src->data + (int)bot * byte_step * src->ncols;


    double iterp_bot, iterp_top;
    if ( right == left )
    {
        iterp_top = top_row[ byte_step * (int) left];
    }
    else
    {
        iterp_top = ( dlx ) * top_row[ byte_step * (int)left ]  
            + ( drx ) * top_row[ byte_step * (int)right ];
    }
    if ( right == left )
    {
        iterp_bot = bot_row[ byte_step * (int) left];
    }
    else
    {
        iterp_bot = ( dlx ) * bot_row[ byte_step * (int)left ]  
            + ( drx ) * bot_row[ byte_step * (int)right ];
    }
    if ( top == bot )
    {
        *dst = iterp_top;
    }
    else
    {
        *dst = ( dby ) * iterp_bot + ( dty ) * iterp_top;
    }
}


bool image_write_rawpbm( char *filename, image_t *im )
{

	FILE *stream = 0;
	
	stream = fopen( filename, "w" );
	if ( !stream )
	{
		fprintf( stderr, "could not open image '%s'\n", filename );
		return false;
	}

    int c;
    int maxval;

    switch (  im->nchan )
    {
        case 1:
            c = '5';
            break;
        case 3:
            c = '6';
            break;
        default:
            fprintf( stderr, "tried saving PBM with unsupported nchan\n" );
            fclose( stream );
            return false;
    }

    maxval = ( 1 << ( im->depth ) ) - 1;

    if ( fprintf( stream, "P%c %d %d %d\n", c, im->ncols, im->nrows, maxval ) < 3 )
    {
        fprintf( stderr, "could not write header for PBM\n" );
        fclose( stream );
        return false;
    }

    int nbytes = im->depth >> 3;

    unsigned int nbytes_to_write = im->nrows * im->ncols * nbytes *im->nchan;
    if ( fwrite(  im->data, 1, nbytes_to_write, stream ) != nbytes_to_write )
    {
        fprintf( stderr, "failed to write all output to file\n" );
        fclose( stream );
        return false;
    } 

    fclose( stream );
    return true;
}


bool is_string( char *s )
{
    for ( unsigned int i = 0; i < strlen( s ); i++ )
    {
        if ( !isprint( s[i] ) )
        {
            return false;
        }
    }
    return true;
}

bool image_read_rawpbm( char *filename, image_t *im )
{
	FILE *stream = 0;
	
	stream = fopen( filename, "r" );
	if ( !stream )
	{
		fprintf( stderr, "could not open image '%s'\n", filename );
		return false;
	}

	char c;
	if ( fscanf( stream, "P%c", &c ) < 1 )
	{
        fprintf( stderr, "magic number in header is wrong from pgm '%s'\n", filename );
        fclose( stream );
        return false;
	}

	int nchan;
	if ( c == '5' )
	{
		nchan = 1;
	}
	else if ( c == '6' )
	{
		nchan = 3;
	}
	else
	{
		fprintf( stderr, "unsupported portable bitmap format P%c\n", c );
		fclose( stream );
		return false;
	}

	char *pch;
	char buf[512];
	int nrows, ncols, maxval;
	bool gotNcols, gotNrows, gotMaxval;
	gotNrows = gotNcols = gotMaxval = false;
	while ( fgets( buf, 512, stream ) != 0 )
	{

		// remove comment
		pch = strpbrk( buf, "#");
		if ( pch )
			*pch = '\0';
 
        pch = strtok( buf, "\r\n\t " );
        while ( pch != NULL )
        {
            if ( !gotNcols )
            {
                if ( sscanf( pch, "%d", &ncols ) == 1 && is_string( pch )  )
                {
                    gotNcols = 1;
                }
            }
            else if ( !gotNrows )
            {
                if ( sscanf( pch, "%d", &nrows ) == 1 && is_string( pch ) )
                {
                    gotNrows = 1;
                }
            }
            else if ( !gotMaxval )
            {
                if ( sscanf( pch, "%d", &maxval ) == 1 && is_string( pch ) )
                {
                    gotMaxval = 1;
                }
            }
            if ( gotNcols && gotNrows && gotMaxval )
                break;
            pch = strtok( NULL, "\r\n\t " );
        }

        if ( gotNcols && gotNrows && gotMaxval )
            break;
	}

	if ( !(gotNcols && gotNrows && gotMaxval) )
	{
		fprintf( stderr, "failed parsing header of '%s'\n", filename );
		fclose( stream );
		return false;
	}
	

	// get depth from maxval
	int depth = 0;
	int t = maxval;
	while ( t )
	{
		t = t >> 1;
		depth++;
	}

	unsigned int total_bytes = nrows * ncols * ( depth >> 3 ) * nchan;
	im->depth = depth;
	im->nrows = nrows;
	im->ncols = ncols;
	im->nchan =  nchan;
	im->data = new unsigned char[ nrows * ncols  * ( depth >> 3 ) * nchan ];

	if ( fread( im->data, 1, total_bytes, stream ) < total_bytes )
	{
		fprintf( stderr, "read less than expected in file '%s'\n", filename );
		fclose( stream );
		return false;
	}


	fclose( stream );
	return true;
}
