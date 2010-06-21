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

bool readRawPbm( char *filename, image_t *im );
bool writeRawPbm( char *filename, image_t *im );
void bilinearInterp( image_t *src, double x, double y, double *dst );

#endif
