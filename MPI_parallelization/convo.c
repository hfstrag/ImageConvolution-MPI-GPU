/*
 * University of Lille
 * Convolution calculation on an image.
 */


#include <stdio.h>
#include <stdlib.h>
#include <math.h>   /* for rint() */
#include <string.h> /* for memcpy() */
#include <sys/time.h>   /* for timing */

#include "rasterfile.h"

#define MAX(a,b) ((a>b) ? a : b)

/** 
 * \struct Raster
 * Structure describing an image in Sun Raster format 
 */

typedef struct {
  struct rasterfile file;                                                ///< Header Sun Raster image 
  unsigned char red[256],green[256],blue[256];  ///< Color palette
  unsigned char *data;                                              ///< Pointer to the image
} Raster;




double my_gettimeofday(){
  struct timeval tmp_time;
  gettimeofday(&tmp_time, NULL);
  return tmp_time.tv_sec + (tmp_time.tv_usec * 1.0e-6L);
}




/**
 * This function converts a LINUX integer to a SUN integer (endianness).
 *
 * \param i: pointer to the integer to convert
 */

void swap(int *i) {
  unsigned char s[4],*n;
  memcpy(s,i,4);
  n=(unsigned char *)i;
  n[0]=s[3];
  n[1]=s[2];
  n[2]=s[1];
  n[3]=s[0];
}

/**
 * \brief Reading an image in Sun Rasterfile format.
 *
 * Upon return from this function, the structure r is filled with the data
 * associated with the image. The r.file field contains the image header
 * information (dimensions, encoding, etc.).
 * The r.data field is a pointer, allocated by the read_rasterfile() function,
 * and which contains the image. This space must be freed after use.
 *
 * \param name: image filename
 * \param r: Raster structure that contains the image loaded into memory
 */

void read_rasterfile(char *name, Raster *r) {
  FILE *f;
  int i;
    
  if( (f=fopen(name, "r"))==NULL) {
    fprintf(stderr,"Error when reading the file %s\n", name);
    exit(1);
  }
  fread(&(r->file), sizeof(struct rasterfile), 1, f);    
  swap(&(r->file.ras_magic));
  swap(&(r->file.ras_width));
  swap(&(r->file.ras_height));
  swap(&(r->file.ras_depth));
  swap(&(r->file.ras_length));
  swap(&(r->file.ras_type));
  swap(&(r->file.ras_maptype));
  swap(&(r->file.ras_maplength));
    
  if ((r->file.ras_depth != 8) ||  (r->file.ras_type != RT_STANDARD) ||
      (r->file.ras_maptype != RMT_EQUAL_RGB)) {
    fprintf(stderr,"Unsuitable palette\n");
    exit(1);
  }
    
  /* component of the palette */
  fread(&(r->red),r->file.ras_maplength/3,1,f);
  fread(&(r->green), r->file.ras_maplength/3,1,f);
  fread(&(r->blue), r->file.ras_maplength/3,1,f);
    
  if ((r->data=malloc(r->file.ras_width*r->file.ras_height))==NULL){
    fprintf(stderr,"Memory allocation error\n");
    exit(1);
  }
  fread(r->data,r->file.ras_width*r->file.ras_height,1,f);
  fclose(f);
}

/**
 * Save an image in Sun Rasterfile format 
 */

void save_rasterfile(char *name, Raster *r)     {
  FILE *f;
  int i;
  
  if( (f=fopen( name, "w"))==NULL) {
    fprintf(stderr,"Error when writing the file %s\n", name);
    exit(1);
  }
    
  swap(&(r->file.ras_magic));
  swap(&(r->file.ras_width));
  swap(&(r->file.ras_height));
  swap(&(r->file.ras_depth));
  swap(&(r->file.ras_length));
  swap(&(r->file.ras_type));
  swap(&(r->file.ras_maptype));
  swap(&(r->file.ras_maplength));
    
  fwrite(&(r->file),sizeof(struct rasterfile),1,f);
  /* component of the palette */
  fwrite(&(r->red),256,1,f);
  fwrite(&(r->green),256,1,f);
  fwrite(&(r->blue),256,1,f);
  /* integer conversion for the image size */
  swap(&(r->file.ras_width));
  swap(&(r->file.ras_height));
  fwrite(r->data,r->file.ras_width*r->file.ras_height,1,f); 
  fclose(f);
}

/**
 * Performs more precise integer division than the '/' operator.
 * Note: The rint function comes from the math library.
 */

unsigned char division(int numerator, int denominator) {  

  if (denominator != 0)
    return (unsigned char) rint((double)numerator/(double)denominator); 
  else 
    return 0;
}

static int order( unsigned char *a, unsigned char *b) {
  return (*a-*b);
}


typedef enum {
  CONVOL_AVG1,      ///< averaging filter
  CONVOL_AVG2,      ///< central averaging filter
  CONVOL_PEAK,       ///< Laplacien
  CONVOL_STEP,       ///< max gradient
  CONVOL_MEDIAN  ///< median filter 
} filter_t;

/**
 * Performs a convolution operation with a predefined kernel on a single point.
 *
 * \param choice: kernel type for convolution:
 * - CONVOL_AVG1: averaging filter
 * - CONVOL_AVG2: averaging filter with a higher central weight
 * - CONVOL_PEAK: edge-extracting filter (Laplacian)
 * - CONVOL_STEP: edge-extracting filter (maximum of gradient components)
 * - CONVOL_MEDIAN: median filter (the 9 values ​​are sorted and the median value is returned)
 * \param NW, N, NE, W, C, E, SW, S, SE: the values ​​of the 9 points involved in the convolution calculation (the convolution is formally a linear combination of these 9 values).
 * \return the convolution value
 */

unsigned char filter( filter_t choice, 
		      unsigned char NW, unsigned char N, unsigned char NE, 
		      unsigned char W, unsigned char C, unsigned char E, 
		      unsigned char SW, unsigned char S, unsigned char SE) {
  int numerator,denominator;

  switch (choice)
    {
    case CONVOL_AVG1:
	  /*  averaging filter */
	  numerator = (int)NW + (int)N + (int)NE + (int)W + (int)C + 
	    (int)E + (int)SW + (int)S + (int)SE;
	  denominator = 9;
	  return division(numerator,denominator); 

    case CONVOL_AVG2:
	  /*  central averaging filter */
	  numerator = (int)NW + (int)N + (int)NE + (int)W + 4*(int)C +
	    (int)E + (int)SW + (int)S + (int)SE;
	  denominator = 12;
	  return division(numerator,denominator);	

    case CONVOL_PEAK:
	  /* peak type edge extraction */
	  numerator = -(int)N - (int)W + 4*(int)C - (int)E - (int)S;
	  /* numerator = -(int)NW -(int)N - (int)NE - (int)W + 8*(int)C -
		 (int)E - (int)SW - (int)S - (int)SE;
	  */
	  return ((4*abs(numerator) > 255) ? 255 :  4*abs(numerator));

    case CONVOL_STEP:
	  /* step type edge extraction */
	  numerator = MAX(abs(C-E),abs(C-S));
	  return ((4*numerator > 255) ? 255 :  4*numerator);

    case CONVOL_MEDIAN:{
          unsigned char tab[] = {NW,N,NE,W,C,E,SW,S,SE};
	  /* non linear filter: quicksort on luminance */
	  qsort(tab, 9, sizeof(unsigned char), (int (*) (const void *,const void *))order);
	  return tab[4];
    }

    default:
	  printf("\nError: unknown filter!\n\n");
	  exit(1);
    }
}

/**
 * Convolution of an image using a predefined filter
 * \param choice: filter selection (see the filter() function)
 * \param tab: pointer to the image
 * \param nbl, nbc: image dimensions
 *
 * \sa filter()
 */

int convolution( filter_t choice, unsigned char tab[], int nbl, int nbc) {
  int i,j;
  unsigned char *tmp;
  
  /* Memory allocation of the intermediate buffer: */
  tmp = (unsigned char*) malloc(sizeof(unsigned char) *nbc*nbl);
  if (tmp == NULL) {
    printf("Error when allocating memory for tmp in convolution()\n");
    return 1;
  }
  
  /* Skipping the edges: */
  for(i=1 ; i<nbl-1 ; i++){
    for(j=1 ; j<nbc-1 ; j++){
      tmp[i*nbc+j] = filter(
			    choice,
			    tab[(i+1)*nbc+j-1],tab[(i+1)*nbc+j],tab[(i+1)*nbc+j+1],
			    tab[(i  )*nbc+j-1],tab[(i)*nbc+j],tab[(i)*nbc+j+1],
			    tab[(i-1)*nbc+j-1],tab[(i-1)*nbc+j],tab[(i-1)*nbc+j+1]);
    } /* for j */
  } /* for i */
  
  /* Copying the processed image back into the original image. 
   * Note that the first and last rows, and first and last columns are not copied (which forces the line-by-line copy). */
  for( i=1; i<nbl-1; i++){
    memcpy( tab+nbc*i+1, tmp+nbc*i+1, (nbc-2)*sizeof(unsigned char));
  } /* for i */
  
  /* Freeing the memory of the intermediate buffer: */
  free(tmp);   
}


/**
 * User interface
 */

static char usage [] = "Usage: %s <SunRaster image name> [0|1|2|3|4] <nbiter>\n";

/*
 * Main part 
 */
#if 0
int main(int argc, char *argv[]) {

  /* Variables relating to the image */
  Raster r;
  int    w, h;	/* number of lines and columns of the image */

  /* Variables related to the image processing */
  int 	 filter;		/* filter number */
  int 	 nbiter;		/* number of iterations */

  /* Variables related to timing */
  double start, end;

  /* Loop variables */
  int 	i,j;


  if (argc != 4) {
    fprintf( stderr, usage, argv[0]);
    return 1;
  }
      
  /* Parameter entry */
  filter = atoi(argv[2]);
  nbiter = atoi(argv[3]);
        
  /* Reading the Raster file */
  read_rasterfile( argv[1], &r);
  h = r.file.ras_height;
  w = r.file.ras_width;
    
  /* Start of timing */
  start = my_gettimeofday();            

  /* The convolution */
  for(i=0 ; i < nbiter ; i++){
    convolution( filter, r.data, h, w);
  } /* for i */

  /* End of timing */
  end = my_gettimeofday();
  printf("Total calculation time: %g second(s) \n", end - start);
    
    /* Saving the Raster file */
  { 
    char output_name[100] = "";
    sprintf(output_name, "post-convolution_filter%d_nbIter%d.ras", filter, nbiter);
    save_rasterfile(output_name, &r);
  }

  return 0;
}
#endif
