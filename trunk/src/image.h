#ifndef IMAGE_H
#define IMAGE_H

typedef struct _image
{
	int nrows, ncols;
	unsigned char *data;
	int depth;
	int nchan;

	_image()
	{
		nrows = 0;
		ncols = 0;
		data = 0;
		depth = 8;
		nchan = 1;
	}


} image_t;

void image_normalize( image_t *src, image_t *dst );
bool image_read_rawpbm( char *filename, image_t *im );
bool image_write_rawpbm( char *filename, image_t *im );
void image_bilinear_interp1chan( image_t *src, double x, double y, double *dst);

#endif
