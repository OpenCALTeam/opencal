#ifndef cal2DRun_h
#define cal2DRun_h

#include "cal2D.cuh"

/*! \brief Structure that defines the cellular automaton's simulation run specifications for Cuda version.
*/
struct CudaCALRun2D
{
	struct CudaCALModel2D* ca2D;	//!< Pointer to the cellular automaton structure.
	struct CudaCALModel2D* device_ca2D;	//!< Pointer to the cellular automaton structure on device.
	struct CudaCALModel2D* h_device_ca2D;	//!< TOCOPY.

	
	//thrust::device_ptr<int> dR; //!< Is a support array for the stream compaction
	//thrust::device_ptr<CALint> array_of_index;
	//unsigned int * device_array_of_index;
	unsigned int * device_array_of_index_dim;


	int step;			//!< Current simulation step.
	int initial_step;	//!< Initial simulation step.
	int final_step;		//!< Final simulation step; if 0 the simulation becomes a loop.

	enum CALUpdateMode UPDATE_MODE;	//!< Callbacks substates' update mode; it can be CAL_UPDATE_EXPLICIT or CAL_UPDATE_IMPLICIT.

	void (*init)(struct CudaCALModel2D*);				//!< Simulation's initialization callback function.
	void (*globalTransition)(struct CudaCALModel2D*);	//!< CA's globalTransition callback function. If defined, it is executed instead of cal2D.c::calGlobalTransitionFunction2D.
	void (*steering)(struct CudaCALModel2D*);			//!< Simulation's steering callback function.
	void (*stopCondition)(struct CudaCALModel2D*);	//!< Simulation's stopCondition callback function.
	void (*finalize)(struct CudaCALModel2D*);			//!< Simulation's finalize callback function.
};

/*! \brief Creates an object of type calRunDef2D, sets its records and returns it as a pointer; it defines the cellular automaton simulation structure for Cuda version.
*/
struct CudaCALRun2D* calCudaRunDef2D(
struct CudaCALModel2D* device_ca2D, //!< Pointer to the cellular automaton structure on device.
struct CudaCALModel2D* ca2D,			//!< Pointer to the cellular automaton structure.
	int initial_step,					//!< Initial simulation step; default value is 0.
	int final_step,					//!< Finale step; if it is 0, a loop is obtained. In order to set final_step to 0, the constant CAL_RUN_LOOP can be used.
	enum CALUpdateMode UPDATE_MODE		//!< Update mode: explicit on or explicit off (implicit).
	);	

/*! \brief Adds a simulation initialization function to CudaCALRun2D for Cuda version.
*/
void calCudaRunAddInitFunc2D(struct CudaCALRun2D* simulation,			//!< Pointer to the run structure.
							 void (*init)(struct CudaCALModel2D*)		//!< Simulation's initialization callback function.
							 );

/*! \brief Adds a CA's globalTransition callback function for CUDA version.
If defined, it is executed instead of cal2D.cu::calCudaGlobalTransitionFunction2D.
*/
void calCudaRunAddGlobalTransitionFunc2D(struct CudaCALRun2D* simulation,					//!< Pointer to the run structure.
										 void (*globalTransition)(struct CudaCALModel2D*)	//!< CA's globalTransition callback function. If defined, it is executed instead of cal2D.c::calGlobalTransitionFunction2D.
										 );

/*! \brief Adds a simulation steering function to CudaCALRun2D for CUDA version.
*/
void calCudaRunAddSteeringFunc2D(struct CudaCALRun2D* simulation,			//!< Pointer to the run structure.
								 void (*steering)(struct CudaCALModel2D*)	//!< Simulation's steering callback function.
								 );

/*! \brief Adds a stop condition function to CALRun2D for CUDA version.
*/
void calCudaRunAddStopConditionFunc2D(struct CudaCALRun2D* simulation,					//!< Pointer to the run structure.
									  void (*stopCondition)(struct CudaCALModel2D*)	//!< Simulation's stopCondition callback function.
									  );

/*! \brief Adds a finalization function to CALRun2D for CUDA version.
*/
void calCudaRunAddFinalizeFunc2D(struct CudaCALRun2D* simulation,			//!< Pointer to the run structure.
								 void (*finalize)(struct CudaCALModel2D*)	//!< Simulation's finalize callback function.
								 );


/*! \brief It executes the simulation initialization function.
*/
void calCudaRunInitSimulation2D(struct CudaCALRun2D* simulation,	//!< Pointer to the run structure.
								dim3 grid, //!< Grid of blocks configuration
								dim3 block //!< Block of threads configuration.
								);

/*! \brief A single step of the cellular automaton. It execute the transition function, the steering and check for the stop condition.
This version is relative for CUDA version.
*/
CALbyte calCudaRunCAStep2D(struct CudaCALRun2D* simulation, //!< Pointer to the run structure.
						   dim3 grid, //!< Grid of blocks configuration
						   dim3 block);//!< Block of threads configuration.

/*! \brief It executes the simulation finalization function.
*/
void calCudaRunFinalizeSimulation2D(struct CudaCALRun2D* simulation	//!< Pointer to the run structure.
									);

/*! \brief Main simulation cicle. It can become a loop is CALRun2D::final_step == 0.
*/
void calCudaRun2D(	struct CudaCALRun2D* simulation,		//!< Pointer to the run structure.
				  dim3 grid, //!< grid of blocks for GPGPU instruction.
				  dim3 block //!< block of threads for GPGPU instruction.
				  );

void calCudaRunFinalize2D(struct CudaCALRun2D* cal2DRun		//!< Pointer to the run structure.
						  );

#endif
