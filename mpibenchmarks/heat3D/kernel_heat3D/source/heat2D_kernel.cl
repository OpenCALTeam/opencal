#include <OpenCAL-CL/calcl3D.h>

#define Q_temperature 0
//model&materials parameters
#define DELTA_X (0.001)
#define DELTA_Y (0.001)
#define DELTA_Z (0.001)
#define DELTA_T (0.001)
#define THERMAL_CONDUCTIVITY (1)
#define MASS_DENSITY (1)
#define SPECIFIC_HEAT_CAPACITY (1)
#define THERMAL_DIFFUSIVITY ( (THERMAL_CONDUCTIVITY)/(SPECIFIC_HEAT_CAPACITY)*(MASS_DENSITY) )
#define THERMAL_DIFFUSIVITY_WATER (1.4563e-4) //C/m^2
#define INIT_TEMP (1200)
#define EPSILON (0.01)

// The cell's transition function (first and only elementary process)
__kernel void heat3D_transitionFunction(__CALCL_MODEL_3D)
{
        //calclThreadCheck3D();

    if( (calclGlobalSlice() < 1) || (calclGlobalSlice() >= calclGetSlices()-1) || get_global_id(0)>=calclGetRows() || get_global_id(1)>=calclGetColumns()|| get_global_id(2)>=calclGetSlices()) return;

        int i = calclGlobalRow();
        int j = calclGlobalColumn();
        int k = calclGlobalSlice();
        const CALreal radius = 5;
        int ROWS = calclGetRows();
        int COLS = calclGetColumns();
        int LAYERS = calclGetSlices();
        int min = (offset == 0)*2;
        int max =1;
        if(offset == 250)
            max = 2;

        if(i > 1 && i < ROWS-1 && j > 1 && j < COLS-1 && k > min && k < LAYERS-max){

                CALreal currValue =calclGet3Dr(MODEL_3D, Q_temperature , i , j , k );

                CALreal dx2 = (calclGet3Dr(MODEL_3D, Q_temperature , i+1,j,k) + calclGet3Dr(MODEL_3D,Q_temperature ,i-1,j,k) - (2*currValue))/(DELTA_X*DELTA_X);


                CALreal dy2 = (calclGet3Dr(MODEL_3D,Q_temperature ,i,j+1,k) + calclGet3Dr(MODEL_3D,Q_temperature ,i,j-1,k) - (2*currValue))/(DELTA_Y*DELTA_Y);


                CALreal dz2 = (calclGet3Dr(MODEL_3D,Q_temperature ,i,j,k+1) + calclGet3Dr(MODEL_3D,Q_temperature ,i,j,k-1) - (2*currValue))/(DELTA_Z*DELTA_Z);

                CALreal newValue = currValue + DELTA_T*THERMAL_DIFFUSIVITY_WATER * (dx2 + dy2 +dz2);
                //||
                if(newValue > EPSILON  && newValue < 10000){
//                    if(calclGetSlices() == 4)
//                    {

//                           printf("%f %f \n", calclGet3Dr(MODEL_3D,Q_temperature ,i,j,k+1), calclGet3Dr(MODEL_3D,Q_temperature ,i,j,k-1));
//                           printf("newVal i,j,k = %i, %i, %i -> dx2=%.15f , dy2=%.15f , dz2=%.15f , val =%.15f \n" ,i,j,k, dx2,dy2,dz2,newValue);
//                    }
                        //printf("newVal i,j,k = %i, %i, %i -> dx2=%.15f , dy2=%.15f , dz2=%.15f , val =%.15f \n" ,i,j,k, dx2,dy2,dz2,newValue);
                        calclSet3Dr(MODEL_3D, Q_temperature, i, j, k, newValue);
                        newValue = currValue;

                }


        }else{
                calclSet3Dr(MODEL_3D, Q_temperature, i, j, k, 0);
        }

            if(i == 1 && j ==1 && k ==1){
//               // printf("calclGetSlices() %d\n", calclGetSlices());

//                if(calclGetSlices() == 5)
//                    printf("offset = % d \n", offset);
//                for (int kk = 0; kk < calclGetSlices(); ++kk) {
//                    for (int ii = 0; ii < 5; ++ii) {
//                        for (int jj = 0; jj < 5; ++jj) {
//                            printf("%f ", calclGet3Dr(MODEL_3D, Q_temperature, ii, jj,kk));
//                        }
//                        printf("\n");
//                    }
//                    printf("\n\n");

//                }
//                printf(" finish \n\n");
//                }

            }


}
