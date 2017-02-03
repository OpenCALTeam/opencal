// Conway's game of Life transition function kernel

#include <OpenCAL-CL/calcl2D.h>




// MODEL KERNEL STARTS HERE -----------------------------------------

#define DEVICE_Q_red (0)
#define DEVICE_Q_green (1)
#define DEVICE_Q_blue (2)



__kernel void sobel2D_transitionFunction(__CALCL_MODEL_2D)
{

	calclThreadCheck2D();
    int i = calclGlobalRow()+borderSize;   
	int j = calclGlobalColumn();
	
	CALint sizeOfX_ = calclGetNeighborhoodSize();
	
	int KX[3][3] = {
						{-1,0,1},
						{-2,0,2},
						{-1,0,1}
					};
					
	int KY[3][3] = {
						{1,2,1},
						{0,0,0},
						{-1,-2,-1}
					};
	
	int Gx=0;
	int Gy=0;
	int n=0;
	int k;
	int k1;
	if(j>0 && j<calclGetColumns()-1)
	for( k = -1; k <= 1 ; k++){
		for( k1 =-1 ; k1 <= 1 ; k1++){
			Gx+=calclGet2Di(MODEL_2D,DEVICE_Q_red, i+k, j+k1)*KX[k+1][k1+1];
			Gy+=calclGet2Di(MODEL_2D,DEVICE_Q_red, i+k, j+k1)*KY[k+1][k1+1];
		}
	}
		
	const double R = (Gx*Gx+Gy*Gy);
	const int P = sqrt(R);
	calclSet2Di(MODEL_2D, DEVICE_Q_red, i, j,P);
	
	return;
	
	


}


//blur kernel
/*__kernel void sobel2D_transitionFunction(__CALCL_MODEL_2D)
{

	calclThreadCheck2D();
    int i = calclGlobalRow()+borderSize;   
	int j = calclGlobalColumn();
	
	CALint sizeOfX_ = calclGetNeighborhoodSize();
	
	int sum=0;
	int n=0;
	for (n = 0; n < sizeOfX_; n++) {
           sum+= calclGetX2Di(MODEL_2D,DEVICE_Q_red, i, j, n);
    }
	 
	 calclSet2Di(MODEL_2D, DEVICE_Q_red, i, j,sum/sizeOfX_);
	


	return;
	
}*/