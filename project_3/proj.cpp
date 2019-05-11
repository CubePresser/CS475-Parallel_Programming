#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <time.h>

FILE* outFile;
unsigned int seed = time(NULL);

// State of the system
int	NowYear;		// 2019 - 2024
int	NowMonth;		// 0 - 11

float	NowPrecip;		// inches of rain per month
float	NowTemp;		// temperature this month
float	NowHeight;		// grain height in inches
int	    NowNumDeer;		// number of deer in the current population

int     NowNumHumans;   // Number of humans, invasive species

// Unit of grain growth in inches
const float GRAIN_GROWS_PER_MONTH =		8.0;
const float ONE_DEER_EATS_PER_MONTH =		0.5;
const float ONE_HUMAN_EATS_GRAIN_PER_MONTH = 0.75;

// Units of precipitation in inches
const float AVG_PRECIP_PER_MONTH =		6.0;	// average
const float AMP_PRECIP_PER_MONTH =		6.0;	// plus or minus
const float RANDOM_PRECIP =			2.0;	// plus or minus noise

// Temperature is in Fahrenheit
const float AVG_TEMP =				50.0;	// average
const float AMP_TEMP =				20.0;	// plus or minus
const float RANDOM_TEMP =			10.0;	// plus or minus noise

const float MIDTEMP =				40.0;
const float MIDPRECIP =				10.0;

float Ranf(unsigned int*, float, float);
int Ranf(unsigned int*, int, int);

// RNG
float Ranf( unsigned int *seedp,  float low, float high )
{
        float r = (float) rand_r( seedp );              // 0 - RAND_MAX

        return(   low  +  r * ( high - low ) / (float)RAND_MAX   );
}

// RNG
int Ranf( unsigned int *seedp, int ilow, int ihigh )
{
        float low = (float)ilow;
        float high = (float)ihigh + 0.9999f;

        return (int)(  Ranf(seedp, low,high) );
}

// Square number
float SQR( float x )
{
        return x*x;
}

// Deer thread
// Computes next number of deer
void GrainDeer() {
    while( NowYear < 2025 )
    {
        // compute a temporary next-value for this quantity
        // based on the current state of the simulation:
        int NextNumDeer = NowNumDeer;

        if(NowNumDeer <= NowNumHumans) {
            NextNumDeer--;
        }

        if(NowNumDeer > NowHeight) {
            NextNumDeer--;
        } else if (NowNumDeer < NowHeight){
            NextNumDeer++;
        }

        if(NowHeight > NowNumHumans) {
            NextNumDeer++;
        }

        // Clamp number of deer to zero
        if(NextNumDeer < 0) {
            NextNumDeer = 0;
        }

        // DoneComputing barrier:
        #pragma omp barrier
        NowNumDeer = NextNumDeer;

        // DoneAssigning barrier:
        #pragma omp barrier

        // DonePrinting barrier:
        #pragma omp barrier
    }
}

// Grain thread
// Computes next grain height
void Grain() {
    while( NowYear < 2025 )
    {
        float NextHeight = NowHeight;
        // compute a temporary next-value for this quantity
        // based on the current state of the simulation:
        float tempFactor = exp(   -SQR(  ( NowTemp - MIDTEMP ) / 10.  )   );
        float precipFactor = exp(   -SQR(  ( NowPrecip - MIDPRECIP ) / 10.  )   );

        NextHeight += tempFactor * precipFactor * GRAIN_GROWS_PER_MONTH;
        NextHeight -= (float)NowNumDeer * ONE_DEER_EATS_PER_MONTH;

        if(NowHeight < NowNumHumans) {
            NextHeight += tempFactor * precipFactor * GRAIN_GROWS_PER_MONTH;
        }

        // Clamp to zero
        if(NextHeight < 0.0) {
            NextHeight = 0.0;
        }

        // DoneComputing barrier:
        #pragma omp barrier
        NowHeight = NextHeight;

        // DoneAssigning barrier:
        #pragma omp barrier

        // DonePrinting barrier:
        #pragma omp barrier
    }
}

// Watcher thread
void Watcher() {
    // Basic timestep will be one month
    while( NowYear < 2025 )
    {
        // DoneComputing barrier:
        #pragma omp barrier
        // DoneAssigning barrier:
        #pragma omp barrier

        printf("========================================\n");
        // Year - Month
        printf("Y: %d\tM:%d\n", NowYear, NowMonth);
        // Precipitation
        printf("Precip: %f\n", NowPrecip);
        // Temperature
        printf("Temp: %f\n", NowTemp);
        // Grain Height
        printf("Grain Height: %f\n", NowHeight);
        // Num Deer
        printf("#Deer: %d\n", NowNumDeer);
        // Num Humans
        printf("#Humans: %d\n", NowNumHumans);
        
        int totalMonths = ((NowYear - 2019) * 12) + NowMonth + 1; 
        fprintf(outFile, "%d\t%f\t%f\t%f\t%d\t%d\n", totalMonths, (5.0/9.0)*(NowTemp-32.0), NowPrecip*2.54, NowHeight*2.54, NowNumDeer, NowNumHumans);

        if(++NowMonth > 11) {
            NowYear++;
            NowMonth = 0;
        }

        float ang = (  30.*(float)NowMonth + 15.  ) * ( M_PI / 180. );

        float temp = AVG_TEMP - AMP_TEMP * cos( ang );
        NowTemp = temp + Ranf( &seed, -RANDOM_TEMP, RANDOM_TEMP );

        float precip = AVG_PRECIP_PER_MONTH + AMP_PRECIP_PER_MONTH * sin( ang );
        NowPrecip = precip + Ranf( &seed,  -RANDOM_PRECIP, RANDOM_PRECIP );
        if( NowPrecip < 0. ) {
            NowPrecip = 0.;
        }

        // DonePrinting barrier:
        #pragma omp barrier
    }
}

// Unique agent thread
void MyAgent() {
    while( NowYear < 2025 )
    {
        // compute a temporary next-value for this quantity
        // based on the current state of the simulation:

        int NextHumans = NowNumHumans;

        if(NowNumHumans < NowNumDeer && NowNumHumans < NowHeight) {
            NextHumans += 3;
        } else if(NowNumHumans < NowHeight) {
            NextHumans += 2;
        } else if(NowNumHumans < NowNumDeer) {
            NextHumans += 1;
        }

        if(NextHumans < 0) {
            NextHumans = 0;
        }

        // DoneComputing barrier:
        #pragma omp barrier
        NowNumHumans = NextHumans;

        // DoneAssigning barrier:
        #pragma omp barrier

        // DonePrinting barrier:
        #pragma omp barrier
    }
}

int main(int argc, char** argv) {
    outFile = fopen("results.txt", "a");
    if(!outFile) {
        printf("Cannot open results.txt for writing\n");
        exit(-1);
    }

    // starting date and time:
    NowMonth =    0;
    NowYear  = 2019;

    // starting state (feel free to change this if you want):
    NowNumDeer = 2;
    NowHeight =  3.;
    NowNumHumans = 1;

    float ang = (  30.*(float)NowMonth + 15.  ) * ( M_PI / 180. );

    float temp = AVG_TEMP - AMP_TEMP * cos( ang );
    NowTemp = temp + Ranf( &seed, -RANDOM_TEMP, RANDOM_TEMP );

    float precip = AVG_PRECIP_PER_MONTH + AMP_PRECIP_PER_MONTH * sin( ang );
    NowPrecip = precip + Ranf( &seed,  -RANDOM_PRECIP, RANDOM_PRECIP );
    if( NowPrecip < 0. ) {
        NowPrecip = 0.;
    }

    omp_set_num_threads( 4 );	// same as # of sections
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            GrainDeer( );
        }

        #pragma omp section
        {
            Grain( );
        }

        #pragma omp section
        {
            Watcher( );
        }

        #pragma omp section
        {
            MyAgent( );	// your own
        }
    }       // implied barrier -- all functions must return in order
        // to allow any of them to get past here
}