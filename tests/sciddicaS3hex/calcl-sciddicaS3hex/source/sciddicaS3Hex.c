#include "sciddicaS3Hex.h"

#include <OpenCAL/cal2DUnsafe.h>
#include <stdlib.h>

//-----------------------------------------------------------------------
//	   THE s3hex(oy) cellular automaton definition section
//-----------------------------------------------------------------------

//cadef and rundef
struct CALModel2D* s3hex;						//the cellular automaton
struct sciddicaS3hexSubstates Q;						//the substates
struct sciddicaS3hexParameters P;						//the parameters
struct CALCLDeviceManager * calcl_device_manager; //the device manager object
struct CALCLModel2D * device_CA;									//the device-side CA


void s3hexRomoveInactiveCells(struct CALModel2D* s3hex, int i, int j)
{
#ifdef ACTIVE_CELLS
    if (calGet2Dr(s3hex,Q.h,i,j) <= P.adh)
        calRemoveActiveCell2D(s3hex,i,j);
#endif
}

//------------------------------------------------------------------------------
//					s3hex simulation functions
//------------------------------------------------------------------------------

void doErosion(struct CALModel2D* s3hex, int i, int j, CALreal	erosion_depth)
{
    CALreal z, d, h, p, runup;

    z = calGet2Dr(s3hex,Q.z,i,j);
    d = calGet2Dr(s3hex,Q.d,i,j);
    h = calGet2Dr(s3hex,Q.h,i,j);
    p = calGet2Dr(s3hex,Q.p,i,j);

    if (h > 0)
        runup =  p/h + erosion_depth;
    else
        runup = erosion_depth;

    calSetCurrent2Dr(s3hex,Q.z,i,j, (z - erosion_depth));
    calSetCurrent2Dr(s3hex,Q.d,i,j, (d - erosion_depth));
    calSet2Dr(s3hex,Q.h,i,j, (h + erosion_depth));
    calSet2Dr(s3hex,Q.p,i,j, (h + erosion_depth)*runup);
}

void sciddicaS3hexSimulationInit(struct CALModel2D* s3hex)
{
    int i, j, n;

    //s3hex parameters setting
    P.adh = P_ADH;
    P.rl = P_RL;
    P.r = P_R;
    P.f = P_F;
    P.mt = P_MT;
    P.pef = P_PEF;
    //P.ltt = P_LTT;

    //initializing substates
    calInitSubstate2Dr(s3hex, Q.h, 0);
    calInitSubstate2Dr(s3hex, Q.p, 0);
    for (n=0; n<s3hex->sizeof_X; n++)
    {
        calInitSubstate2Dr(s3hex, Q.fh[n], 0);
        calInitSubstate2Dr(s3hex, Q.fp[n], 0);
    }

#ifdef ACTIVE_CELLS
    for (i=0; i<s3hex->rows; i++)
        for (j=0; j<s3hex->columns; j++){
            int d = calGet2Dr(s3hex,Q.d,i,j);
            if (d > 0)
            {
                int s = calGet2Di(s3hex,Q.s,i,j);
                if (s == -1) {
                    calSetCurrent2Di(s3hex,Q.s,i,j,0);
                    doErosion(s3hex,i,j,d);
                    calAddActiveCell2D(s3hex,i,j);
                }
            }
        }
#endif
    calUpdateActiveCells2D(s3hex);
}

//------------------------------------------------------------------------------
//					s3hex CADef and runDef
//------------------------------------------------------------------------------

void sciddicaS3hexCADef()
{
    int n;
    CALbyte optimization_type = CAL_NO_OPT;

#ifdef ACTIVE_CELLS
    optimization_type = CAL_OPT_ACTIVE_CELLS_NAIVE;
#endif

    //cadef and rundef
    s3hex = calCADef2D (ROWS, COLS, CAL_HEXAGONAL_NEIGHBORHOOD_2D, CAL_SPACE_TOROIDAL, optimization_type);

    //add substates
    for (n=0; n<s3hex->sizeof_X; n++)
    {
        Q.fh[n] = calAddSubstate2Dr(s3hex);
    }
    for (n=0; n<s3hex->sizeof_X; n++)
    {
        Q.fp[n] = calAddSubstate2Dr(s3hex);
    }
    Q.z = calAddSingleLayerSubstate2Dr(s3hex);
    Q.d = calAddSingleLayerSubstate2Dr(s3hex);
    Q.s = calAddSingleLayerSubstate2Di(s3hex);
    Q.h = calAddSubstate2Dr(s3hex);
    Q.p = calAddSubstate2Dr(s3hex);


    //load configuration
    sciddicaS3hexLoadConfig();
    sciddicaS3hexSimulationInit(s3hex);
}


void sciddicaS3hexCALCLDef(){

#ifdef ACTIVE_CELLS
    char * kernelSrc = KERNEL_SRC_AC;
    char * kernelInc = KERNEL_INC_AC;
#else
    char * kernelSrc = KERNEL_SRC;
    char * kernelInc = KERNEL_INC;
#endif

    CALCLdevice device;
    CALCLcontext context;
    CALCLprogram program;
    //OpenCL device selection from stdin and context definition
    calcl_device_manager = calclCreateManager();
    calclGetPlatformAndDeviceFromStdIn(calcl_device_manager, &device);
    context = calclCreateContext(&device);

    // Load kernels and return a compiled program
    program = calclLoadProgram2D(context, device, kernelSrc, kernelInc);

    //device-side CA definition
    device_CA = calclCADef2D(s3hex, context, program, device);

    CALCLkernel kernel_s3hexErosion;
    CALCLkernel kernel_s3hexFlowsComputation;
    CALCLkernel kernel_s3hexWidthAndPotentialUpdate;
    CALCLkernel kernel_s3hexClearOutflows;
    CALCLkernel kernel_s3hexEnergyLoss;

    kernel_s3hexErosion = calclGetKernelFromProgram(&program, KERNEL_S3HEXEROSION);

    kernel_s3hexFlowsComputation = calclGetKernelFromProgram(&program, KERNEL_S3HEXFLOWSCOMPUTATION);

    kernel_s3hexWidthAndPotentialUpdate = calclGetKernelFromProgram(&program, KERNEL_S3HEXWIDTHANDPOTENTIALUPDATE);

    kernel_s3hexClearOutflows = calclGetKernelFromProgram(&program, KERNEL_S3HEXCLEAROUTFLOWS);

    kernel_s3hexEnergyLoss = calclGetKernelFromProgram(&program, KERNEL_S3HEXENERGYLOSS);

#ifdef ACTIVE_CELLS
    CALCLkernel kernel_s3hexRemoveInactiveCells = calclGetKernelFromProgram(&program, KERNEL_S3HEXROMOVEINACTIVECELLS);
#endif

    // Register transition function’s elementary processes to the device-side CA
    calclAddElementaryProcess2D(device_CA, &kernel_s3hexErosion);
    calclAddElementaryProcess2D(device_CA, &kernel_s3hexFlowsComputation);
    calclAddElementaryProcess2D(device_CA, &kernel_s3hexWidthAndPotentialUpdate);
    calclAddElementaryProcess2D(device_CA, &kernel_s3hexClearOutflows);
    calclAddElementaryProcess2D(device_CA, &kernel_s3hexEnergyLoss);
#ifdef ACTIVE_CELLS
    calclAddElementaryProcess2D(device_CA, &kernel_s3hexRemoveInactiveCells);
#endif
    time_t start_time, end_time;

    start_time = time(NULL);
    calclRun2D(device_CA, 1, STEPS);
    end_time = time(NULL);
    printf("%ld", end_time - start_time);
    sciddicaS3hexSaveConfig();
}

//------------------------------------------------------------------------------
//					s3hex I/O functions
//------------------------------------------------------------------------------

void sciddicaS3hexLoadConfig()
{
    //load configuration
    calLoadSubstate2Dr(s3hex, Q.z, DEM_PATH);
    calLoadSubstate2Dr(s3hex, Q.d, REGOLITH_PATH);
    calLoadSubstate2Di(s3hex, Q.s, SOURCE_PATH);
}

void sciddicaS3hexSaveConfig()
{
    calSaveSubstate2Dr(s3hex, Q.h, OUTPUT_PATH);
}

//------------------------------------------------------------------------------
//					s3hex finalization function
//------------------------------------------------------------------------------


void sciddicaS3hexExit()
{
    //finalizations
    calFinalize2D(s3hex);
    calclFinalizeManager(calcl_device_manager);
    calclFinalize2D(device_CA);
}
