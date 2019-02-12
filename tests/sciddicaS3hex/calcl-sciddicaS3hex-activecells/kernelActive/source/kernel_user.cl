#include <kernel.h>

void doErosion(__CALCL_MODEL_2D, int i, int j, CALreal	erosion_depth)
{

    CALreal z, d, h, p, runup;
    z = calclGetSingleLayer2Dr(MODEL_2D,Z,i,j);
    d = calclGetSingleLayer2Dr(MODEL_2D,D,i,j);
    h = calclGet2Dr(MODEL_2D,H,i,j);
    p = calclGet2Dr(MODEL_2D,P,i,j);

    if (h > 0)
        runup =  p/h + erosion_depth;
    else
        runup = erosion_depth;

    calclSetSingleLayer2Dr(MODEL_2D,Z,i,j, (z - erosion_depth));
    calclSetSingleLayer2Dr(MODEL_2D,D,i,j, (d - erosion_depth));
    calclSet2Dr(MODEL_2D,H,i,j, (h + erosion_depth));
    calclSet2Dr(MODEL_2D,P,i,j, (h + erosion_depth)*runup);
}

__kernel void s3hexErosion(__CALCL_MODEL_2D)
{
    calclActiveThreadCheck2D();

    int threadID = calclGlobalRow();
    int i = calclActiveCellRow(threadID);
    int j = calclActiveCellColumn(threadID);

    CALint s;
    CALreal d, p, erosion_depth;


    d = calclGetSingleLayer2Dr(MODEL_2D,D,i,j);


    if (d > 0)
    {

        s = calclGetSingleLayer2Di(MODEL_2D,S,i,j);

        if (s <  -1)
            calclSetSingleLayer2Di(MODEL_2D, S, i, j, s+1);

        if (s == -1) {
            calclSetSingleLayer2Di(MODEL_2D, S, i, j, 0);
            doErosion(MODEL_2D,i,j,d);
            calclAddActiveCell2D(MODEL_2D, i, j);

        }

        p = calclGet2Dr(MODEL_2D,P,i,j);
        if (p > P_MT) {
            erosion_depth = p * P_PEF;
            if (erosion_depth > d)
                erosion_depth = d;
            doErosion(MODEL_2D,i,j,erosion_depth);
        }

    }

}

__kernel void s3hexFlowsComputation(__CALCL_MODEL_2D)
{

    calclActiveThreadCheck2D();

    int threadID = calclGlobalRow();
    int i = calclActiveCellRow(threadID);
    int j = calclActiveCellColumn(threadID);

    CALbyte eliminated_cells[NUMBER_OF_OUTFLOWS]={CAL_FALSE,CAL_FALSE,CAL_FALSE,CAL_FALSE,CAL_FALSE, CAL_FALSE, CAL_FALSE};
    CALbyte again;
    CALint cells_count;
    CALreal average;
    CALreal m;
    CALreal u[NUMBER_OF_OUTFLOWS], delta_H[NUMBER_OF_OUTFLOWS], delta_z[NUMBER_OF_OUTFLOWS];
    CALint n;
    CALreal z_0, h_0, z_n, h_n, runup_0, z_0_plus_runup_0, sum;
    CALreal h, p, h0_out, p0_out ;


    if (calclGet2Dr(MODEL_2D,H,i,j) <= P_ADH)
        return;



    z_0 = calclGetSingleLayer2Dr(MODEL_2D, Z, i, j);
    h_0 = calclGet2Dr(MODEL_2D, H, i, j);
    runup_0 = calclGet2Dr(MODEL_2D, P, i, j) / h_0;
    z_0_plus_runup_0 = z_0 + runup_0;

    m = runup_0;
    u[0] = z_0;
    delta_z[0] = 0;
    delta_H[0] = 0;

    CALint sizeOf_X = calclGetNeighborhoodSize();

    for (n=1; n<sizeOf_X; n++)
    {
        z_n = calclGetSingleLayerX2Dr(MODEL_2D, Z, i, j, n);
        h_n = calclGetX2Dr(MODEL_2D, H, i, j, n);

        u[n] = z_n + h_n;
        delta_z[n] = z_0 - z_n;
        delta_H[n] = z_0_plus_runup_0 - u[n];
    }

    for (n=1; n<sizeOf_X; n++)
        eliminated_cells[n] = (delta_H[n] < P_F);
    //computes outflows
    do{
        again = CAL_FALSE;
        average = m;
        cells_count = 0;

        for (n=0; n<sizeOf_X; n++)
            if (!eliminated_cells[n]){
                average += u[n];
                cells_count++;
            }

        if (cells_count != 0)
            average /= cells_count;

        for (n=0; n<sizeOf_X; n++)
            if( (average<=u[n]) && (!eliminated_cells[n]) ){
                eliminated_cells[n]=CAL_TRUE;
                again=CAL_TRUE;
            }

    }while (again);


    sum = 0;
    for (n=0; n<sizeOf_X; n++)
        if (!eliminated_cells[n])
            sum += average - u[n];

    h0_out = p0_out = 0;
    for (n=1; n<sizeOf_X; n++)
        if (!eliminated_cells[n])
        {
            h = h_0 * ((average-u[n])/sum) * P_R;
            p = (z_0_plus_runup_0 - u[n])*h;

            h0_out += h;
            p0_out += runup_0*h;

            calclSet2Dr(MODEL_2D, n, i, j, h);
            calclSet2Dr(MODEL_2D, n+NUMBER_OF_OUTFLOWS, i, j, p);
            calclAddActiveCellX2D(MODEL_2D, i, j, n);
        }

    if (h0_out > 0)
        calclSet2Dr(MODEL_2D, FH0, i, j, h0_out);
    if (p0_out > 0)
        calclSet2Dr(MODEL_2D, FP0, i, j, p0_out);
}


__kernel void s3hexWidthAndPotentialUpdate(__CALCL_MODEL_2D)
{
    calclActiveThreadCheck2D();

    int threadID = calclGlobalRow();
    int i = calclActiveCellRow(threadID);
    int j = calclActiveCellColumn(threadID);

    CALint sizeOf_X = calclGetNeighborhoodSize();

    CALreal h_next, p_next;
    CALint n, m;

    h_next = calclGet2Dr(MODEL_2D, H, i, j) - calclGet2Dr(MODEL_2D, FH0, i, j);

    for(n=1; n<sizeOf_X; n++)
    {
        if (n <= 3) m = n+3; else m = n-3;
        h_next +=  calclGetX2Dr(MODEL_2D, m, i, j, n);
    }
    calclSet2Dr(MODEL_2D, H, i, j, h_next);

    p_next = calclGet2Dr(MODEL_2D, P, i, j) - calclGet2Dr(MODEL_2D, FP0, i, j);
    for(n=1; n<sizeOf_X; n++)
    {
        if (n <= 3) m = n+3; else m = n-3;
        p_next +=  calclGetX2Dr(MODEL_2D, m+NUMBER_OF_OUTFLOWS, i, j, n);
    }
    calclSet2Dr(MODEL_2D, P, i, j, p_next);
}

__kernel void s3hexClearOutflows(__CALCL_MODEL_2D)
{
    calclActiveThreadCheck2D();

    int threadID = calclGlobalRow();
    int i = calclActiveCellRow(threadID);
    int j = calclActiveCellColumn(threadID);

    CALint sizeOf_X = calclGetNeighborhoodSize();
    int n;
    for (n=0; n<sizeOf_X; n++)
    {
        if (calclGet2Dr(MODEL_2D, n, i, j) > 0.0)
            calclSet2Dr(MODEL_2D, n, i, j, 0.0);
        if (calclGet2Dr(MODEL_2D, n+NUMBER_OF_OUTFLOWS, i, j) > 0.0)
            calclSet2Dr(MODEL_2D, n+NUMBER_OF_OUTFLOWS, i, j, 0.0);
    }
}


__kernel void s3hexEnergyLoss(__CALCL_MODEL_2D)
{
    calclActiveThreadCheck2D();

    int threadID = calclGlobalRow();
    int i = calclActiveCellRow(threadID);
    int j = calclActiveCellColumn(threadID);

    CALreal h, runup;

    if (calclGet2Dr(MODEL_2D,H,i,j) <= P_ADH)
        return;

    h = calclGet2Dr(MODEL_2D,H,i,j);
    if (h > P_ADH) {
        runup = calclGet2Dr(MODEL_2D,P,i,j) / h - P_RL;
        if (runup < h)
            runup = h;
        calclSet2Dr(MODEL_2D,P,i,j,h*runup);
    }
}

__kernel void s3hexRemoveInactiveCells(__CALCL_MODEL_2D)
{
    calclActiveThreadCheck2D();

    int threadID = calclGlobalRow();
    int i = calclActiveCellRow(threadID);
    int j = calclActiveCellColumn(threadID);

    if (calclGet2Dr(MODEL_2D, H, i ,j) <= P_ADH)
      calclRemoveActiveCell2D(MODEL_2D, i , j);
}
