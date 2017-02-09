// Conway's game of Life transition function kernel

#include <OpenCAL-CL/calcl2D.h>
 
#define DEVICE_Q_temperature (0)
#define DEVICE_Q_material (1)

#define SIZE (500)
#define ROWS (SIZE)
#define COLS (SIZE)
#define MATERIAL_START_ROW (ROWS/2-ROWS/8)
#define SOURCE_SIZE (20)
#define MATERIAL_END_ROW (MATERIAL_START_ROW+SOURCE_SIZE)



#define EPSILON (0.01)

//model&materials parameters
#define DELTA_X (0.001)
#define DELTA_Y (0.001)

#define DELTA_T (0.001)

#define THERMAL_CONDUCTIVITY (1)
#define MASS_DENSITY (1)
#define SPECIFIC_HEAT_CAPACITY (1)
#define THERMAL_DIFFUSIVITY ( (THERMAL_CONDUCTIVITY)/(SPECIFIC_HEAT_CAPACITY)*(MASS_DENSITY) )
#define THERMAL_DIFFUSIVITY_WATER (1.2563e-5) //C/m^2

#define THERMAL_CONDUCTIVITY_SOLID (1)
#define MASS_DENSITY_SOLID (1)
#define SPECIFIC_HEAT_CAPACITY_SOLID (1)
//#define THERMAL_DIFFUSIVITY_SOLID ( (THERMAL_CONDUCTIVITY_SOLID)/(SPECIFIC_HEAT_CAPACITY_SOLID)*(MASS_DENSITY_SOLID) )
#define THERMAL_DIFFUSIVITY_SOLID (1.8563e-4) //C/m^2


#define INIT_TEMP (100)


__kernel void heat3D_transitionFunction(__CALCL_MODEL_2D)
{


	calclThreadCheck2D();
    int i = calclGlobalRow()+borderSize;   
	int j = calclGlobalColumn();
	
		
	if(i > 1 && i < ROWS-1 && j > 1 && j < COLS-1 ){
		
		CALreal currValue =calclGet2Dr(MODEL_2D, DEVICE_Q_temperature , i , j );

		CALreal dx2 = (calclGet2Dr(MODEL_2D, DEVICE_Q_temperature , i+1 , j ) + 
						calclGet2Dr(MODEL_2D, DEVICE_Q_temperature , i-1, j ) 
						- (2*currValue))/(DELTA_X*DELTA_X);


		CALreal dy2 = (calclGet2Dr(MODEL_2D, DEVICE_Q_temperature , i , j+1 ) 
						+ calclGet2Dr(MODEL_2D, DEVICE_Q_temperature , i , j-1 ) 
						- (2*currValue))/(DELTA_Y*DELTA_Y);
	

		CALbyte material = calclGet2Db(MODEL_2D, DEVICE_Q_material , i,j);
        CALreal newValue = 0.0f;
		
		if(material)
            newValue= currValue + DELTA_T*THERMAL_DIFFUSIVITY_SOLID * (dx2 + dy2 );    
        
        else
            newValue = currValue + DELTA_T*THERMAL_DIFFUSIVITY_WATER * (dx2 + dy2 );
			
		if(newValue > EPSILON  && newValue < 10000){
			//xprintf("newVal i,j,k = %i, %i, %i -> dx2=%.15f , dy2=%.15f , dz2=%.15f , val =%.15f \n" ,i,j,k, dx2,dy2,dz2,newValue);
			calclSet2Dr(MODEL_2D, DEVICE_Q_temperature, i, j, newValue);
			

		}
	}
	
	 if(i-1 > MATERIAL_START_ROW && i-1<MATERIAL_END_ROW && j>=225 && j<=250){
		calclSet2Dr(MODEL_2D, DEVICE_Q_temperature, i, j, INIT_TEMP);
			
		
	 }
	



}
