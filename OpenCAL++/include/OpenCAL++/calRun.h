//
// Created by knotman on 14/04/16.
//

#ifndef OPENCAL_ALL_CALRUN_H
#define OPENCAL_ALL_CALRUN_H


#include <OpenCAL++/calModel.h>

namespace opencal {

template< class MODEL>
class CALRun {

protected:
    typedef  MODEL CALMODEL_type;
    typedef  MODEL* CALMODEL_pointer;
    typedef  MODEL& CALMODEL_reference;


    CALMODEL_pointer calModel;

    int step;			//!< Current simulation step.
    int initial_step;	//!< Initial simulation step.
    int final_step;		//!< Final simulation step; if 0 the simulation becomes a loop.

    enum calCommon :: CALUpdateMode UPDATE_MODE;	//!< Callbacks substates' update mode; it can be CAL_UPDATE_EXPLICIT or CAL_UPDATE_IMPLICIT.

    bool initFunctionRedefined = true;
    bool globalTransitionFunctionRedefined = true;
    bool steeringFunctionRedefined = true;
    bool finalizeFunctionRedefined = true;



public:

    /*! \brief CALRun's constructor, it defines the cellular automaton simulation structure.
        */
    CALRun (CALMODEL_pointer _calModel,			//!< Pointer to the cellular automaton structure.
            int _initial_step,					//!< Initial simulation step; default value is 0.
            int _final_step,					//!< Finale step; if it is 0, a loop is obtained. In order to set final_step to 0, the constant CAL_RUN_LOOP can be used.
            enum calCommon :: CALUpdateMode _UPDATE_MODE		//!< Update mode: explicit on or explicit off (implicit).
            ): UPDATE_MODE(_UPDATE_MODE) , calModel(_calModel), initial_step(_initial_step), final_step(_final_step), step(_initial_step)
    {


    }


    /*! \brief CALRun's destructor.
        */
    ~CALRun (){
    }

    inline virtual void init ()
    {
        initFunctionRedefined = false;
    }

    inline virtual void steering ()
    {
        steeringFunctionRedefined = false;
    }

    inline virtual void finalize ()
    {
        finalizeFunctionRedefined = false;
    }

    inline virtual void global ()
    {
        globalTransitionFunctionRedefined = false;
    }


    inline virtual bool stopCondition ()
    {
        return false;
    }

    /*! \brief It executes the simulation initialization function.
    */
    void runInitSimulation(){
        init();
        if (initFunctionRedefined)
        {

            if (this->UPDATE_MODE == calCommon :: CAL_UPDATE_IMPLICIT)
                this->calModel->update();
        }
    }


    /*! \brief A single step of the cellular automaton. It executes the transition function, the steering and check for the stop condition.
    */
    calCommon::CALbyte runCAStep(){
        this->global();
        if (this->globalTransitionFunctionRedefined){

            if (this->UPDATE_MODE == calCommon :: CAL_UPDATE_IMPLICIT)
                this->calModel->update();
        }
        else{
            this->calModel->globalTransitionFunction();

        }

        this->steering ();


        if (steeringFunctionRedefined)
            if (this->UPDATE_MODE == calCommon :: CAL_UPDATE_IMPLICIT)
                this->calModel->update();

        return !this->stopCondition();
    }

    /*! \brief It executes the simulation finalization function.
    */
    void runFinalizeSimulation(){
        finalize();
        if (finalizeFunctionRedefined)
        {

            if (this->UPDATE_MODE == calCommon :: CAL_UPDATE_IMPLICIT)
                this->calModel->update();
        }
    }


    /*! \brief Main simulation cicle. It can become a loop is CALRun::final_step == 0.
    */
    void run(){
        calCommon :: CALbyte again = CAL_FALSE;
        runInitSimulation();

        for (this->step = this->initial_step; (this->final_step == CAL_RUN_LOOP || this->step <= this->final_step ); this->step++)
        {
            again = runCAStep();
            if (!again)
                break;
        }

        runFinalizeSimulation();
    }




    int getStep() const
    {
        return step;
    }


};


} //namespace opencal

#endif //OPENCAL_ALL_CALRUN_H
