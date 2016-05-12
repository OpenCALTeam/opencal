//
// Created by knotman on 14/04/16.
//

#ifndef OPENCAL_ALL_CALRUN_H
#define OPENCAL_ALL_CALRUN_H


#include <OpenCAL++/calModel.h>

namespace opencal {

    template<class MODEL, class RET_TYPE>
    class CALModelFunctor{
    public:
        virtual RET_TYPE run(MODEL*)=0;
        CALModelFunctor(){}
        virtual RET_TYPE operator()(MODEL* model){
            return this->run(model);
        }
        virtual ~CALModelFunctor(){}

    };


    template< class MODEL>
    class CALRun {

        typedef  MODEL CALMODEL_type;
        typedef  MODEL* CALMODEL_pointer;
        typedef  MODEL& CALMODEL_reference;

        using VoidFunctor = CALModelFunctor<CALMODEL_type,void>;
        using BoolFunctor = CALModelFunctor<CALMODEL_type,calCommon::CALbyte>;


    protected:
        CALMODEL_pointer calModel;

        int step;			//!< Current simulation step.
        int initial_step;	//!< Initial simulation step.
        int final_step;		//!< Final simulation step; if 0 the simulation becomes a loop.

        enum calCommon :: CALUpdateMode UPDATE_MODE;	//!< Callbacks substates' update mode; it can be CAL_UPDATE_EXPLICIT or CAL_UPDATE_IMPLICIT.


        using InitFunction              = VoidFunctor;
        using GlobalTransitionFunction  = VoidFunctor;
        using SteeringFunction          = VoidFunctor;
        using StopConditionFunction     = BoolFunctor;
        using FinalizeFunction          = VoidFunctor;

        InitFunction* init;								//!< Simulation's initialization callback functor.
        GlobalTransitionFunction* globalTransition;		//!< CA's globalTransition callback function. If defined, it is executed instead of CALModel::globalTransitionFunction.
        SteeringFunction* steering;						//!< Simulation's steering callback function.
        FinalizeFunction* finalize;						//!< Simulation's finalize callback function.

        StopConditionFunction* stopCondition;			//!< Simulation's stopCondition callback function.



    public:

        /*! \brief CALRun's constructor, it defines the cellular automaton simulation structure.
        */
        CALRun (CALMODEL_pointer _calModel,			//!< Pointer to the cellular automaton structure.
                int _initial_step,					//!< Initial simulation step; default value is 0.
                int _final_step,					//!< Finale step; if it is 0, a loop is obtained. In order to set final_step to 0, the constant CAL_RUN_LOOP can be used.
                enum calCommon :: CALUpdateMode _UPDATE_MODE		//!< Update mode: explicit on or explicit off (implicit).
        ): UPDATE_MODE(_UPDATE_MODE) , init(nullptr) , globalTransition(nullptr) , steering(nullptr),
           stopCondition(nullptr) , finalize(nullptr), calModel(_calModel), initial_step(_initial_step), final_step(_final_step), step(_initial_step)
        {


        }


        /*! \brief CALRun's destructor.
        */
        ~CALRun (){
            delete init;
            delete globalTransition;
            delete steering;
            delete stopCondition;
            delete finalize;
        }

        /*! \brief Adds a simulation initialization function to CALRun.
        */
        void addInitFunc(InitFunction* _init 		//!< Simulation's initialization callback function.
        ){
            this->init = _init;
        }


        /*! \brief Adds a CA's globalTransition callback function.
        If defined, it is executed instead of CALModel::globalTransitionFunction.
    */
        void addGlobalTransitionFunc(GlobalTransitionFunction* _globalTransition	//!< CA's globalTransition callback function. If defined, it is executed instead of CALModel::globalTransitionFunction.
        ){
            this->globalTransition = _globalTransition;
        }

        /*! \brief Adds a simulation steering function to CALRun.
    */
        void addSteeringFunc(SteeringFunction*	_steering //!< Simulation's steering callback function.
        ){
            this->steering = _steering;
        }


        /*! \brief Adds a stop condition function to CALRun.
    */
        void addStopConditionFunc(StopConditionFunction* _stopCondition	//!< Simulation's stopCondition callback function.
        ){
            this->stopCondition = _stopCondition;
        }

        /*! \brief Adds a finalization function to CALRun.
        */
        void addFinalizeFunc(FinalizeFunction* _finalize	//!< Simulation's finalize callback function.
        ){
            this->finalize = _finalize;
        }

        /*! \brief It executes the simulation initialization function.
    */
        void runInitSimulation(){
            if (this->init != nullptr)
            {
                this->init->run(this->calModel);
                if (this->UPDATE_MODE == calCommon :: CAL_UPDATE_IMPLICIT)
                    this->calModel->update();
            }
        }


        /*! \brief A single step of the cellular automaton. It executes the transition function, the steering and check for the stop condition.
    */
        calCommon::CALbyte runCAStep(){
            if (this->globalTransition!= nullptr){
                this->globalTransition->run(this->calModel);
                if (this->UPDATE_MODE == calCommon :: CAL_UPDATE_IMPLICIT)
                    this->calModel->update();
            }
            else{
                this->calModel->globalTransitionFunction();

            }


            if (this->steering != nullptr){ //No explicit substates and active cells updates are needed in this case

                this->steering->run(this->calModel);
                if (this->UPDATE_MODE == calCommon :: CAL_UPDATE_IMPLICIT)
                    this->calModel->update();
            }

            if (this->stopCondition != nullptr && this->stopCondition->run(this->calModel))
                return CAL_FALSE;

            return CAL_TRUE; //continue simulation only if stopcondition is not set or return false (i.e. not stop)
        }

        /*! \brief It executes the simulation finalization function.
    */
        void runFinalizeSimulation(){
            if (this->finalize!= nullptr)
            {
                this->finalize->run(this->calModel);
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
