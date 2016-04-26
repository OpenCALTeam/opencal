/*
 * Copyright (c) 2016 OpenCALTeam (https://github.com/OpenCALTeam),
 * Telesio Research Group,
 * Department of Mathematics and Computer Science,
 * University of Calabria, Italy.
 *
 * This file is part of OpenCAL (Open Computing Abstraction Layer).
 *
 * OpenCAL is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of
 * the License, or (at your option) any later version.
 *
 * OpenCAL is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with OpenCAL. If not, see <http://www.gnu.org/licenses/>.
 */

#include ".\include\cal2D.cuh"
#include ".\include\cal2DIO.cuh"
#include ".\include\cal2DRun.cuh"
#include ".\include\cal2DToolkit.cuh"
#include ".\include\cal2DBuffer.cuh"
#include ".\include\cal2DBufferIO.cuh"

#include <stdlib.h>
#include <time.h>

#include "cuda_profiler_api.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//-----------------------------------------------------------------------
//   THE Sciara-FV2 CELLULAR AUTOMATON
//-----------------------------------------------------------------------
#define maximum_steps 				0
#define stopping_threshold			0.00
#define refreshing_step            	0
#define thickness_visual_threshold 	0.00
#define Pclock                     	60.00
#define PTsol                      	1143.00
#define PTvent                     	1360.00
#define Pr_Tsol                 	0.0750
#define Pr_Tvent	              	0.90
#define Phc_Tsol	             	60.00
#define Phc_Tvent	            	0.4
#define Pcool                      	9.0
#define Prho                       	2600.00
#define Pepsilon                  	0.90
#define Psigma                     	5.68E-08
#define Pcv                        	1150
#define algorithm                  	MIN
#define layers						40
#define rows						378
#define cols						517
#define cell_size					10.000000
#define nodata_value				0
#define num_emission_rate			15
#define num_total_emission_rates	2
#define xllcorner					499547.500000
#define yllcorner					4174982.500000
#define rad2						1.41421356237

__device__ CALreal a,b,c,d;

#define STEPS 500

#define DEM_PATH "data/2006/2006_000000000000_Morphology.txt"
#define VENTS_PATH "data/2006/2006_000000000000_Vents.txt"
#define EMISSION_RATE_PATH "data/2006/2006_000000000000_EmissionRate.txt"
#define TEMPERATURE_PATH "data/2006/2006_000000000000_Temperature.txt"
#define THICKNESS_PATH "data/2006/2006_000000000000_Thickness.txt"
#define SOLIDIFIED_LAVA_THICKNESS_PATH "data/2006/2006_000000000000_SolidifiedLavaThickness.txt"
#define REAL_EVENT_THICKNESS_PATH "data/2006/2006_000000000000_RealEvent.txt"

#define O_DEM_PATH "data/2006_SAVE/2006_000000000000_Morphology.txt"
#define O_VENTS_PATH "data/2006_SAVE/2006_000000000000_Vents.txt"
#define O_EMISSION_RATE_PATH "data/2006_SAVE/2006_000000000000_EmissionRate.txt"
#define O_TEMPERATURE_PATH "data/2006_SAVE/2006_000000000000_Temperature.txt"
#define O_THICKNESS_PATH "data/2006_SAVE/2006_000000000000_Thickness.txt"
#define O_SOLIDIFIED_LAVA_THICKNESS_PATH "data/2006_SAVE/2006_000000000000_SolidifiedLavaThickness.txt"


#define ACTIVE_CELLS

#define NUMBER_OF_OUTFLOWS 8
#define MOORE_NEIGHBORS 9
#define VON_NEUMANN_NEIGHBORS 5

#define NUMBER_OF_SUBSTATES_REAL 13
#define NUMBER_OF_SUBSTATES_INT 1
#define NUMBER_OF_SUBSTATES_BYTE 1

#define ncols_str "ncols"
#define nrows_str "nrows"
#define xllcorner_str "xllcorner"
#define yllcorner_str "yllcorner"
#define cell_size_str "cellsize"
#define NODATA_value_str "NODATA_value"

enum SUBSTATES_NAMES_REAL{
	ALTITUDE=0,THICKNESS,TEMPERATURE,PRE_EVENT_TOPOGRAPHY, SOLIDIFIED, FLOWN,FLOWO,FLOWE,FLOWS, FLOWNO, FLOWSO, FLOWSE,FLOWNE
};
enum SUBSTATES_NAMES_INT{
	VENTS=0,
};
enum SUBSTATES_NAMES_BYTE{
	TOPOGRAPHY_BOUND=0,
};
CALint N = 21;
CALint M = 47;
dim3 block(N,M);
dim3 grid(cols/block.x, rows/block.y);

__device__
	double thickness() {
		//printf("Pac e Pt: %f %f\n", cell_size*cell_size, Pclock);
		return 1.580890 / (cell_size*cell_size * Pclock);
}
// START FUNCTIONS
__global__
	void updateVentsEmission(struct CudaCALModel2D * model) {

		CALint offset = calCudaGetIndex(model), i = calCudaGetIndexRow(model, offset), j= calCudaGetIndexColumn(model, offset);
		CALreal emitted_lava; //= thickness(); // - is a temporary value, because OpenCAL-CUDA haven't emission_rate funcionality yet.

		//printf("index %d %d %d\n", model->activecell_size_current, i, j);
		if(calCudaGet2Di(model,offset,VENTS) == 1)
		{
			emitted_lava = 1.806732;//1.580890 / (cell_size*cell_size * Pclock); //thickness();
			//printf("Emitted lava %f\n", emitted_lava);
			if (emitted_lava > 0) {
				calCudaSet2Dr(model, offset, calCudaGet2Dr(model, offset, THICKNESS) + emitted_lava, THICKNESS);
				calCudaSet2Dr(model, offset, PTvent, TEMPERATURE);
			}
		}

		if(calCudaGet2Di(model,offset,VENTS) == 2)
		{
			emitted_lava = 1.806732;//0.948534;//1.580890 / (cell_size*cell_size * Pclock); //thickness();

			if (emitted_lava > 0) {
				calCudaSet2Dr(model, offset, calCudaGet2Dr(model, offset, THICKNESS) + emitted_lava, THICKNESS);
				calCudaSet2Dr(model, offset, PTvent, TEMPERATURE);
			}
		}

}

__device__
	double powerLaw(double k1, double k2, double T) {
		double log_value = k1 + k2 * T;
		return pow(10.0, log_value);
}

__device__
	void outflowsMin(struct CudaCALModel2D * model, int offset, CALreal *f) {

		bool n_eliminated[MOORE_NEIGHBORS];
		double z[MOORE_NEIGHBORS];
		double h[MOORE_NEIGHBORS];
		double H[MOORE_NEIGHBORS];
		double theta[MOORE_NEIGHBORS];
		double w[MOORE_NEIGHBORS];		//Distances between central and adjecent cells
		double Pr_[MOORE_NEIGHBORS];		//Relaiation rate arraj
		bool loop;
		int counter;
		double avg, _w, _Pr, hc, sum, sumZ;

		CALreal t = calCudaGet2Dr(model, offset, TEMPERATURE);

		_w = cell_size;
		_Pr = powerLaw(a, b, t);
		hc = powerLaw(c, d, t);

		for (int k = 0; k < MOORE_NEIGHBORS; k++) {

			h[k] = calCudaGetX2Dr(model, offset, k, THICKNESS);
			H[k] = f[k] = theta[k] = 0;
			w[k] = _w;
			Pr_[k] = _Pr;
			CALreal sz = calCudaGetX2Dr(model, offset, k, ALTITUDE);
			CALreal sz0 = calCudaGet2Dr(model, offset, ALTITUDE);
			if (k < VON_NEUMANN_NEIGHBORS)
				z[k] = calCudaGetX2Dr(model, offset, k, ALTITUDE);
			else
				z[k] = sz0 - (sz0 - sz) / rad2;
		}

		//if(calCudaGetIndexRow(cols, offset) == 114 && calCudaGetIndexColumn(cols, offset) == 71)
		//printf("Value: %f %f %f %f\n", h[0], H[0], w[0], z[0]);


		H[0] = z[0];
		n_eliminated[0] = true;

		for (int k = 1; k < MOORE_NEIGHBORS; k++)
			if (z[0] + h[0] > z[k] + h[k]) {
				H[k] = z[k] + h[k];
				theta[k] = atan(((z[0] + h[0]) - (z[k] + h[k])) / w[k]);

				n_eliminated[k] = true;
			} else
				n_eliminated[k] = false;

			do {
				loop = false;
				avg = h[0];
				counter = 0;
				for (int k = 0; k < MOORE_NEIGHBORS; k++)
					if (n_eliminated[k]) {
						avg += H[k];
						counter++;
					}
					if (counter != 0)
						avg = avg / double(counter);
					for (int k = 0; k < MOORE_NEIGHBORS; k++)
						if (n_eliminated[k] && avg <= H[k]) {
							n_eliminated[k] = false;
							loop = true;
						}
			} while (loop);

			for (int k = 1; k < MOORE_NEIGHBORS; k++) {
				if (n_eliminated[k] && h[0] > hc * cos(theta[k])) {
					f[k] = Pr_[k] * (avg - H[k]);
				}
			}

			//if(calCudaGetIndexRow(cols, offset) == 114 && calCudaGetIndexColumn(cols, offset) == 71)
			//	printf("%f %f %f %f \n%f %f %f %f\n\n", f[0],f[1],f[2],f[3],f[4],
			//	f[5],f[6],f[7]);

}

__global__
	void empiricalFlows(struct CudaCALModel2D * model) {

		CALint offset = calCudaGetIndex(model);

		if (calCudaGet2Dr(model, offset, THICKNESS) > 0) {
			CALreal f[MOORE_NEIGHBORS];
			outflowsMin(model, offset, f);

			if (f[1] > 0) {
				calCudaSet2Dr(model, offset, f[1],  FLOWN);
#ifdef ACTIVE_CELLS
				calCudaAddActiveCellX2D(model, offset, 1);
#endif
			}

			if (f[2] > 0) {
				calCudaSet2Dr(model, offset, f[2],  FLOWO);
#ifdef ACTIVE_CELLS
				calCudaAddActiveCellX2D(model, offset, 2);
#endif
			}

			if (f[3] > 0) {
				calCudaSet2Dr(model, offset, f[3],  FLOWE);
#ifdef ACTIVE_CELLS
				calCudaAddActiveCellX2D(model, offset, 3);
#endif
			}

			if (f[4] > 0) {
				calCudaSet2Dr(model, offset, f[4],  FLOWS);
#ifdef ACTIVE_CELLS
				calCudaAddActiveCellX2D(model, offset, 4);
#endif
			}

			if (f[5] > 0) {
				calCudaSet2Dr(model, offset, f[5],  FLOWNO);
#ifdef ACTIVE_CELLS
				calCudaAddActiveCellX2D(model, offset, 5);
#endif
			}

			if (f[6] > 0) {
				calCudaSet2Dr(model, offset, f[6],  FLOWSO);
#ifdef ACTIVE_CELLS
				calCudaAddActiveCellX2D(model, offset, 6);
#endif
			}

			if (f[7] > 0) {
				calCudaSet2Dr(model, offset, f[7],  FLOWSE);
#ifdef ACTIVE_CELLS
				calCudaAddActiveCellX2D(model, offset, 7);
#endif
			}

			if (f[8] > 0) {
				calCudaSet2Dr(model, offset, f[8],  FLOWNE);
#ifdef ACTIVE_CELLS
				calCudaAddActiveCellX2D(model, offset, 8);
#endif
			}

		}
}

__global__
	void width_update(struct CudaCALModel2D* model) {
		CALint outFlowsIndexes[NUMBER_OF_OUTFLOWS] = { 3, 2, 1, 0, 6, 7, 4, 5 };
		CALint n;
		CALint offset = calCudaGetIndex(model);
		CALreal initial_h = calCudaGet2Dr(model, offset, THICKNESS);
		CALreal initial_t = calCudaGet2Dr(model, offset, TEMPERATURE);
		CALreal residualTemperature = initial_h * initial_t;
		CALreal residualLava = initial_h;
		CALreal h_next = initial_h;
		CALreal t_next;

		CALreal ht = 0;
		CALreal inSum = 0;
		CALreal outSum = 0;

		CALreal inFlow = 0, outFlow = 0, neigh_t = 0;

		//for (n = 1; n < model->sizeof_X; n++)
		// n = 1
		inFlow = calCudaGetX2Dr(model, offset, 1, FLOWS);
		outFlow = calCudaGet2Dr(model, offset, FLOWN);
		neigh_t = calCudaGetX2Dr(model, offset, 1, TEMPERATURE);
		ht += inFlow * neigh_t;
		inSum += inFlow;
		outSum += outFlow;

		// n = 2
		inFlow = calCudaGetX2Dr(model, offset, 2, FLOWE);
		outFlow = calCudaGet2Dr(model, offset, FLOWO);
		neigh_t = calCudaGetX2Dr(model, offset, 2, TEMPERATURE);
		ht += inFlow * neigh_t;
		inSum += inFlow;
		outSum += outFlow;

		// n = 3
		inFlow = calCudaGetX2Dr(model, offset, 3, FLOWO);
		outFlow = calCudaGet2Dr(model, offset, FLOWE);
		neigh_t = calCudaGetX2Dr(model, offset, 3, TEMPERATURE);
		ht += inFlow * neigh_t;
		inSum += inFlow;
		outSum += outFlow;

		// n = 4
		inFlow = calCudaGetX2Dr(model, offset, 4, FLOWN);
		outFlow = calCudaGet2Dr(model, offset, FLOWS);
		neigh_t = calCudaGetX2Dr(model, offset, 4, TEMPERATURE);
		ht += inFlow * neigh_t;
		inSum += inFlow;
		outSum += outFlow;

		// n = 5
		inFlow = calCudaGetX2Dr(model, offset, 5, FLOWSE);
		outFlow = calCudaGet2Dr(model, offset, FLOWNO);
		neigh_t = calCudaGetX2Dr(model, offset, 5, TEMPERATURE);
		ht += inFlow * neigh_t;
		inSum += inFlow;
		outSum += outFlow;

		// n = 6
		inFlow = calCudaGetX2Dr(model, offset, 6, FLOWNE);
		outFlow = calCudaGet2Dr(model, offset, FLOWSO);
		neigh_t = calCudaGetX2Dr(model, offset, 6, TEMPERATURE);
		ht += inFlow * neigh_t;
		inSum += inFlow;
		outSum += outFlow;

		// n = 7
		inFlow = calCudaGetX2Dr(model, offset, 7, FLOWNO);
		outFlow = calCudaGet2Dr(model, offset, FLOWSE);
		neigh_t = calCudaGetX2Dr(model, offset, 7, TEMPERATURE);
		ht += inFlow * neigh_t;
		inSum += inFlow;
		outSum += outFlow;

		// n = 8
		inFlow = calCudaGetX2Dr(model, offset, 8, FLOWSO);
		outFlow = calCudaGet2Dr(model, offset, FLOWNE);
		neigh_t = calCudaGetX2Dr(model, offset, 8, TEMPERATURE);
		ht += inFlow * neigh_t;
		inSum += inFlow;
		outSum += outFlow;

		h_next += inSum - outSum;
		calCudaSet2Dr(model, offset, h_next, THICKNESS);
		if (inSum > 0 || outSum > 0) {
			residualLava -= outSum;
			t_next = (residualLava * initial_t + ht) / (residualLava + inSum);
			calCudaSet2Dr(model, offset, t_next, TEMPERATURE);
		}
}

__global__
	void updateTemperature(struct CudaCALModel2D* model) {
		CALreal nT, h, T, aus;
		CALint offset = calCudaGetIndex(model);
		CALreal sh = calCudaGet2Dr(model, offset, THICKNESS);
		CALreal st = calCudaGet2Dr(model, offset, TEMPERATURE);
		CALreal sz = calCudaGet2Dr(model, offset, ALTITUDE);

		if (sh > 0 && !calCudaGet2Db(model, offset, TOPOGRAPHY_BOUND)) {
			h = sh;
			T = st;
			if (h != 0) {
				//			nT = T / h;
				nT = T;

				/*nT -= Pepsilon * Psigma * pow(nT, 4.0) * Pclock * Pcool/ (Prho * Pcv * h * Pac);
				nSt[x][y]=nT;*/

				aus = 1.0 + (3 * pow(nT, 3.0) * Pepsilon * Psigma * Pclock * Pcool) / (Prho * Pcv * h * cell_size * cell_size);
				st = nT / pow(aus, 1.0 / 3.0);
				calCudaSet2Dr(model, offset, st, TEMPERATURE);

			}

			//solidification
			if (st <= PTsol && sh > 0) {
				calCudaSet2Dr(model, offset, sz + sh, ALTITUDE);
				calCudaSetCurrent2Dr(model, offset, calCudaGet2Dr(model, offset, SOLIDIFIED) + sh, SOLIDIFIED);
				calCudaSet2Dr(model, offset, 0, THICKNESS);
				calCudaSet2Dr(model, offset, PTsol, TEMPERATURE);

			} else
				calCudaSet2Dr(model, offset, sz, ALTITUDE);
		}
}

__global__
	void removeActiveCells(struct CudaCALModel2D* model) {
		CALint offset = calCudaGetIndex(model);
		CALreal st = calCudaGet2Dr(model, offset, TEMPERATURE);
		if (st <= PTsol && !calCudaGet2Db(model, offset, TOPOGRAPHY_BOUND))
			calCudaRemoveActiveCell2D(model, offset);
}

__global__
	void stopCondition(struct CudaCALModel2D* model) {
		/*	if (sciara->elapsed_time >= sciara->effusion_duration)
		return CAL_TRUE;

		//La simulazione può continuare
		return CAL_FALSE; */
}

__global__
	void steering(struct CudaCALModel2D* model) {

		CALint offset = calCudaGetIndex(model);

		calCudaInit2Dr(model, offset, 0, FLOWN);
		calCudaInit2Dr(model, offset, 0, FLOWO);
		calCudaInit2Dr(model, offset, 0, FLOWE);
		calCudaInit2Dr(model, offset, 0, FLOWS);
		calCudaInit2Dr(model, offset, 0, FLOWNO);
		calCudaInit2Dr(model, offset, 0, FLOWSO);
		calCudaInit2Dr(model, offset, 0, FLOWSE);
		calCudaInit2Dr(model, offset, 0, FLOWNE);

		if (calCudaGet2Db(model, offset, TOPOGRAPHY_BOUND) == CAL_TRUE) {
			calCudaSet2Dr(model, offset, 0, THICKNESS);
			calCudaSet2Dr(model, offset, 0, TEMPERATURE);
		}

		//elapsed_time += Pclock;

		//updateVentsEmission(model);
}

__device__
	void evaluatePowerLawParams(CALreal value_sol, CALreal value_vent, CALreal &k1, CALreal &k2) {
		k2 = (log10(value_vent) - log10(value_sol)) / (PTvent - PTsol);
		k1 = log10(value_sol) - k2 * (PTsol);
}

__device__
	void MakeBorder(CudaCALModel2D* model) {

		CALint i, j, offset = calCudaGetIndex(model);

		i = calCudaGetIndexRow(model, offset);
		j = calCudaGetIndexColumn(model, offset);

		//prima riga
		if(i == 0){
			if (calCudaGet2Dr(model, offset, ALTITUDE) >= 0) {
				calCudaSetCurrent2Db(model, offset, CAL_TRUE, TOPOGRAPHY_BOUND);
#ifdef ACTIVE_CELLS
				calCudaAddActiveCell2D(model, offset);
#endif
			}
		}

		//ultima riga
		if(i == rows - 1){

			if (calCudaGet2Dr(model, offset, ALTITUDE) >= 0) {
				calCudaSetCurrent2Db(model, offset, CAL_TRUE, TOPOGRAPHY_BOUND);
#ifdef ACTIVE_CELLS
				calCudaAddActiveCell2D(model, offset);
#endif
			}


		}


		//prima colonna
		if( j == 0 ){
			if (calCudaGet2Dr(model, offset, ALTITUDE) >= 0) {
				calCudaSetCurrent2Db(model, offset, CAL_TRUE, TOPOGRAPHY_BOUND);
#ifdef ACTIVE_CELLS
				calCudaAddActiveCell2D(model, offset);
#endif
			}
		}


		//ultima colonna
		if(j == cols - 1){
			if (calCudaGet2Dr(model, offset, ALTITUDE) >= 0) {
				calCudaSetCurrent2Db(model, offset, CAL_TRUE, TOPOGRAPHY_BOUND);
#ifdef ACTIVE_CELLS
				calCudaAddActiveCell2D(model, offset);
#endif
			}

		}

		if( i > 0 && j > 0 && i < rows - 1 && j < cols - 1){
			if (calCudaGet2Dr(model, offset, ALTITUDE) >= 0) {
				for (int k = 1; k < model->sizeof_X; k++)
					if (calCudaGetX2Dr(model, offset, k, ALTITUDE) < 0) {
						calCudaSetCurrent2Db(model, offset, CAL_TRUE, TOPOGRAPHY_BOUND);
#ifdef ACTIVE_CELLS
						calCudaAddActiveCell2D(model, offset);
#endif
						break;
					}
			}
		}

}

__global__
	void simulationInitialize(struct CudaCALModel2D* model) {

		//dichiarazioni
		unsigned int maximum_number_of_emissions = 0;

		//azzeramento dello step dell'AC
		//sciara->step = 0;
		//sciara->elapsed_time = 0;

		//determinazione numero massimo di passi
		//for (unsigned int i = 0; i < sciara->emission_rate.size(); i++)
		//	if (maximum_number_of_emissions < sciara->emission_rate[i].size())
		//		maximum_number_of_emissions = sciara->emission_rate[i].size();
		//maximum_steps_from_emissions = (int)(emission_time/Pclock*maximum_number_of_emissions);
		//sciara->effusion_duration = sciara->emission_time * maximum_number_of_emissions;

		CALint offset = calCudaGetSimpleOffset();

		//calCudaInit2Dr(model, offset, 0, ALTITUDE);
		calCudaInit2Dr(model, offset, 0.000000, THICKNESS);
		//calCudaInit2Dr(model, offset, 0, TEMPERATURE);

		//TODO single layer initialization
		calCudaInit2Db(model, offset, CAL_FALSE, TOPOGRAPHY_BOUND);

		//calCudaInit2Dr(model, offset, 0, SOLIDIFIED);
		//calCudaInit2Dr(model, offset, 0, PRE_EVENT_TOPOGRAPHY);

		calCudaInit2Dr(model, offset, 0, FLOWN);
		calCudaInit2Dr(model, offset, 0, FLOWO);
		calCudaInit2Dr(model, offset, 0, FLOWE);
		calCudaInit2Dr(model, offset, 0, FLOWS);
		calCudaInit2Dr(model, offset, 0, FLOWNO);
		calCudaInit2Dr(model, offset, 0, FLOWSO);
		calCudaInit2Dr(model, offset, 0, FLOWSE);
		calCudaInit2Dr(model, offset, 0, FLOWNE);

		//definisce il bordo della morfologia
		MakeBorder(model);

		//calcolo a b (parametri viscosità) c d (parametri resistenza al taglio)
		evaluatePowerLawParams(Pr_Tsol, Pr_Tvent, a, b);
		evaluatePowerLawParams(Phc_Tsol, Phc_Tvent, c, d);
		//	updateVentsEmission(model);
#ifdef ACTIVE_CELLS
		//printf("Ciao %d\n", offset);
		if(calCudaGet2Di(model,offset,VENTS) == 1){
			//printf("Aggiungo il primo cratere\n");
			calCudaAddActiveCell2D(model, offset);
		}
		if(calCudaGet2Di(model,offset,VENTS) == 2){
			//printf("Aggiungo il secondo cratere\n");
			calCudaAddActiveCell2D(model, offset);
		}
#endif


}

// END FUNCTIONS

int main()
{
	time_t start_time, end_time;
	cudaProfilerStart();

	//cadef and rundef
	struct CudaCALModel2D* sciara_fv2;
	struct CudaCALRun2D* simulation_sciara_fv2;

#ifdef ACTIVE_CELLS
	sciara_fv2 = calCudaCADef2D (rows, cols, CAL_MOORE_NEIGHBORHOOD_2D, CAL_SPACE_TOROIDAL, CAL_OPT_ACTIVE_CELLS);
#else
	sciara_fv2 = calCudaCADef2D (rows, cols, CAL_MOORE_NEIGHBORHOOD_2D, CAL_SPACE_TOROIDAL, CAL_NO_OPT);
#endif

	struct CudaCALModel2D* device_sciara_fv2 = calCudaAlloc();

	//add transition function's elementary processes
	calCudaAddElementaryProcess2D(sciara_fv2, updateVentsEmission);
	calCudaAddElementaryProcess2D(sciara_fv2, empiricalFlows);
	calCudaAddElementaryProcess2D(sciara_fv2, width_update);
	calCudaAddElementaryProcess2D(sciara_fv2, updateTemperature);

#ifdef ACTIVE_CELLS
	calCudaAddElementaryProcess2D(sciara_fv2, removeActiveCells);
#endif

	//add substates
	calCudaAddSubstate2Dr(sciara_fv2,NUMBER_OF_SUBSTATES_REAL);
	calCudaAddSubstate2Di(sciara_fv2,NUMBER_OF_SUBSTATES_INT);
	calCudaAddSubstate2Db(sciara_fv2,NUMBER_OF_SUBSTATES_BYTE);

	//load configuration
	calCudaLoadSubstate2Dr(sciara_fv2, DEM_PATH, ALTITUDE);
	calCudaLoadSubstate2Di(sciara_fv2, VENTS_PATH, VENTS);
	//EMISSION_RATE_MISSED TEMP 10.
	//calCudaLoadSubstate2Dr(sciara_fv2, THICKNESS_PATH, THICKNESS);
	calCudaLoadSubstate2Dr(sciara_fv2, TEMPERATURE_PATH, TEMPERATURE);
	calCudaLoadSubstate2Dr(sciara_fv2, SOLIDIFIED_LAVA_THICKNESS_PATH, SOLIDIFIED);

	calInitializeInGPU2D(sciara_fv2,device_sciara_fv2);

	cudaErrorCheck("Data initialized on device\n");

	simulation_sciara_fv2 = calCudaRunDef2D(device_sciara_fv2, sciara_fv2, 1, STEPS, CAL_UPDATE_IMPLICIT);

	//simulation run
	calCudaRunAddInitFunc2D(simulation_sciara_fv2, simulationInitialize);
	calCudaRunAddSteeringFunc2D(simulation_sciara_fv2, steering);
	calCudaRunAddStopConditionFunc2D(simulation_sciara_fv2, stopCondition);

	printf ("Starting simulation...\n");
	start_time = time(NULL);
	calCudaRun2D(simulation_sciara_fv2, grid, block);

	//send data to CPU
	calSendDataGPUtoCPU(sciara_fv2,device_sciara_fv2);

	cudaErrorCheck("Final configuration sent to CPU\n");
	end_time = time(NULL);
	printf ("Simulation terminated.\nElapsed time: %d\n", end_time-start_time);

	//saving configuration
	calCudaSaveSubstate2Dr(sciara_fv2, O_DEM_PATH, ALTITUDE);
	calCudaSaveSubstate2Dr(sciara_fv2, O_SOLIDIFIED_LAVA_THICKNESS_PATH, SOLIDIFIED);
	calCudaSaveSubstate2Dr(sciara_fv2, O_TEMPERATURE_PATH, TEMPERATURE);
	calCudaSaveSubstate2Dr(sciara_fv2, O_THICKNESS_PATH, THICKNESS);
	calCudaSaveSubstate2Di(sciara_fv2, O_VENTS_PATH, VENTS);

	cudaErrorCheck("Data saved on output file\n");

	//finalizations
	calCudaRunFinalize2D(simulation_sciara_fv2);
	calCudaFinalize2D(sciara_fv2, device_sciara_fv2);
	cudaProfilerStop();
	system("pause");
	return 0;
}
