// Array multiplication: C = A * B:

// System includes
#include <stdio.h>
#include <assert.h>
#include <malloc.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include "helper_functions.h"
#include "helper_cuda.h"


#ifndef BLOCKSIZE
#define BLOCKSIZE		32		// number of threads per block
#endif

#ifndef NUMTRIALS
#define NUMTRIALS		16384		// to make the timing more accurate
#endif

#ifndef TOLERANCE
#define TOLERANCE		0.00001f	// tolerance to relative error
#endif

// ranges for the random numbers:
const float XCMIN =	 0.0;
const float XCMAX =	 2.0;
const float YCMIN =	 0.0;
const float YCMAX =	 2.0;
const float RMIN  =	 0.5;
const float RMAX  =	 2.0;

// function prototypes:
float		Ranf( float, float );
int		    Ranf( int, int );
void		TimeOfDaySeed( );

// array multiplication (CUDA Kernel) on the device: C = A * B

__global__  void ArrayMul( float *XC, float *YC, float *RS, float *C )
{
	__shared__ int hits[BLOCKSIZE];

	unsigned int numItems = blockDim.x;
	unsigned int tnum = threadIdx.x;
	unsigned int wgNum = blockIdx.x;
	unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;

	// randomize the location and radius of the circle:
	float xc = XC[gid];
	float yc = YC[gid];
	float r  = RS[gid];

	// solve for the intersection using the quadratic formula:
	float a = 2.;
	float b = -2.*( xc + yc );
	float c = xc*xc + yc*yc - r*r;
	float d = b*b - 4.*a*c;

	// CASE A: Circle is not completely missed
	if(d >= 0) {

		// hits the circle:
		// get the first intersection:
		d = sqrt( d );
		float t1 = (-b + d ) / ( 2.*a );	// time to intersect the circle
		float t2 = (-b - d ) / ( 2.*a );	// time to intersect the circle
		float tmin = t1 < t2 ? t1 : t2;		// only care about the first intersection

		// CASE B: Circle does not completely engulf the laser pointer
		if(tmin >= 0) {

			// where does it intersect the circle?
			float xcir = tmin;
			float ycir = tmin;

			// get the unitized normal vector at the point of intersection:
			float nx = xcir - xc;
			float ny = ycir - yc;
			float n = sqrt( nx*nx + ny*ny );
			nx /= n;	// unit vector
			ny /= n;	// unit vector

			// get the unitized incoming vector:
			float inx = xcir - 0.;
			float iny = ycir - 0.;
			float in = sqrt( inx*inx + iny*iny );
			inx /= in;	// unit vector
			iny /= in;	// unit vector

			// get the outgoing (bounced) vector:
			float dot = inx*nx + iny*ny;
			float outy = iny - 2.*ny*dot;	// angle of reflection = angle of incidence`

			// find out if it hits the infinite plate:
			float t = ( 0. - ycir ) / outy;

			// CASE C (false): Reflected beam went up instead of down
			// CASE D (true): Beam hit the infinite plate
			if(t >= 0) {
				hits[tnum] = 1.;
			} else {
				hits[tnum] = 0.;
			}
		} else {
			hits[tnum] = 0.;
		}
	} else {
		hits[tnum] = 0.;
	}

	for (int offset = 1; offset < numItems; offset *= 2)
	{
		int mask = 2 * offset - 1;
		__syncthreads();
		if ((tnum & mask) == 0)
		{
			hits[tnum] += hits[tnum + offset];
		}
	}

	__syncthreads();
	if (tnum == 0)
		C[wgNum] = hits[0];
}


// main program:

int
main( int argc, char* argv[ ] )
{
	int dev = findCudaDevice(argc, (const char **)argv);

	TimeOfDaySeed( );		// seed the random number generator

	FILE* outfile = fopen("results.txt", "a");
    if(!outfile) {
        fprintf(stderr, "Error opening results.txt\n");
        return -1;
    }

	// allocate host memory:

	float * hXC = new float [ NUMTRIALS ];
	float * hYC = new float [ NUMTRIALS ];
	float * hRS = new float [ NUMTRIALS ];
	float * hC  = new float [ NUMTRIALS / BLOCKSIZE ];

	// fill the random-value arrays:
    for( int n = 0; n < NUMTRIALS; n++ )
    {       
        hXC[n] = Ranf( XCMIN, XCMAX );
        hYC[n] = Ranf( YCMIN, YCMAX );
		hRS[n] = Ranf(  RMIN,  RMAX );
	}
	
	for( int n = 0; n < NUMTRIALS / BLOCKSIZE; n++) {
		hC[n]  = 0.;
	}

	// allocate device memory:

	float *dXC, *dYC, *dRS, *dC;

	dim3 dimsXC( NUMTRIALS, 1, 1 );
	dim3 dimsYC( NUMTRIALS, 1, 1 );
	dim3 dimsRS( NUMTRIALS, 1, 1 );
	dim3 dimsC( NUMTRIALS/BLOCKSIZE, 1, 1 );

	//__shared__ float prods[SIZE/BLOCKSIZE];


	cudaError_t status;
	status = cudaMalloc( reinterpret_cast<void **>(&dXC), NUMTRIALS*sizeof(float) );
		checkCudaErrors( status );
	status = cudaMalloc( reinterpret_cast<void **>(&dYC), NUMTRIALS*sizeof(float) );
		checkCudaErrors( status );
	status = cudaMalloc( reinterpret_cast<void **>(&dRS), NUMTRIALS*sizeof(float) );
		checkCudaErrors( status );
	status = cudaMalloc( reinterpret_cast<void **>(&dC), (NUMTRIALS/BLOCKSIZE)*sizeof(float) );
		checkCudaErrors( status );


	// copy host memory to the device:

	status = cudaMemcpy( dXC, hXC, NUMTRIALS*sizeof(float), cudaMemcpyHostToDevice );
		checkCudaErrors( status );
	status = cudaMemcpy( dYC, hYC, NUMTRIALS*sizeof(float), cudaMemcpyHostToDevice );
		checkCudaErrors( status );
	status = cudaMemcpy( dRS, hRS, NUMTRIALS*sizeof(float), cudaMemcpyHostToDevice );
		checkCudaErrors( status );

	// setup the execution parameters:

	dim3 threads(BLOCKSIZE, 1, 1 );
	dim3 grid( NUMTRIALS / threads.x, 1, 1 );

	// Create and start timer

	cudaDeviceSynchronize( );

	// allocate CUDA events that we'll use for timing:

	cudaEvent_t start, stop;
	status = cudaEventCreate( &start );
		checkCudaErrors( status );
	status = cudaEventCreate( &stop );
		checkCudaErrors( status );

	// record the start event:

	status = cudaEventRecord( start, NULL );
		checkCudaErrors( status );

	// execute the kernel:

	ArrayMul<<< grid, threads >>>( dXC, dYC, dRS, dC );

	// record the stop event:

	status = cudaEventRecord( stop, NULL );
		checkCudaErrors( status );

	// wait for the stop event to complete:

	status = cudaEventSynchronize( stop );
		checkCudaErrors( status );

	float msecTotal = 0.0f;
	status = cudaEventElapsedTime( &msecTotal, start, stop );
		checkCudaErrors( status );

	// compute and print the performance

	double secondsTotal = 0.001 * (double)msecTotal;
	double trialsPerSecond = (float)NUMTRIALS / secondsTotal;
	double megaTrialsPerSecond = trialsPerSecond / 1000000.;
	fprintf( stderr, "Number of trials = %10d, MegaTrials/Second = %10.2lf\n", NUMTRIALS, megaTrialsPerSecond );
	fprintf(outfile, "%f\t", megaTrialsPerSecond);

	// copy result from the device to the host:

	status = cudaMemcpy( hC, dC, (NUMTRIALS/BLOCKSIZE)*sizeof(float), cudaMemcpyDeviceToHost );
		checkCudaErrors( status );

	// check the sum :

	int numHits = 0.;
	for(int i = 0; i < NUMTRIALS/BLOCKSIZE; i++ )
	{
		//printf("Hits at %d:\t%d\n", i, (int)hC[i]);
		numHits += (int)hC[i];
	}
	printf("Numhits: %d\n", numHits);
	fprintf( stderr, "\nprobability = %10.2lf \n", (double)numHits / (double)NUMTRIALS );

	// clean up memory:
	delete [ ] hXC;
	delete [ ] hYC;
	delete [ ] hRS;
	delete [ ] hC;

	status = cudaFree( dXC );
		checkCudaErrors( status );
	status = cudaFree( dYC );
		checkCudaErrors( status );
	status = cudaFree( dRS );
		checkCudaErrors( status );
	status = cudaFree( dC );
		checkCudaErrors( status );


	return 0;
}

float
Ranf( float low, float high )
{
    float r = (float) rand();               // 0 - RAND_MAX
    float t = r  /  (float) RAND_MAX;       // 0. - 1.

    return   low  +  t * ( high - low );
}

int
Ranf( int ilow, int ihigh )
{
    float low = (float)ilow;
    float high = ceil( (float)ihigh );

    return (int) Ranf(low,high);
}

void
TimeOfDaySeed( )
{
    struct tm y2k = { 0 };
    y2k.tm_hour = 0;   y2k.tm_min = 0; y2k.tm_sec = 0;
    y2k.tm_year = 100; y2k.tm_mon = 0; y2k.tm_mday = 1;

    time_t  timer;
    time( &timer );
    double seconds = difftime( timer, mktime(&y2k) );
    unsigned int seed = (unsigned int)( 1000.*seconds );    // milliseconds
    srand( seed );
}
