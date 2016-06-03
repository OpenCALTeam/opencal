#ifndef calModelUnsafe_h
#define calModelUnsafe_h

#include <OpenCAL++/calModel.h>
#include <OpenCAL++/calSubstateUnsafe.h>
namespace opencal {
    template<uint DIMENSION, class NEIGHBORHOOD, typename COORDINATE_TYPE = uint>
    class   CALModelUnsafe: public CALModel<DIMENSION,NEIGHBORHOOD, COORDINATE_TYPE>
    {
    private:
         typedef NEIGHBORHOOD* NEIGHBORHOOD_pointer;
        /*! \brief Substates allocation function.
        */

        template<class PAYLOAD, calCommon::SUBSTATE_OPT OPT= calCommon::NO_OPT>
        calCommon::CALbyte allocSubstateUnsafe(CALSubstate<PAYLOAD , DIMENSION , COORDINATE_TYPE, OPT>*& Q){
            using BUFFER = CALBuffer<PAYLOAD, DIMENSION, COORDINATE_TYPE>;
            using SUBSTATE = CALSubstateUnsafe<PAYLOAD, DIMENSION , COORDINATE_TYPE, OPT>;

            BUFFER* current = new BUFFER(this->coordinates);
            BUFFER* next = new BUFFER(this->coordinates);
            Q = new SUBSTATE (current, next, this->coordinates);

            if (!Q->getCurrent() || !Q->getNext()){
                return CAL_FALSE;
            }

            return CAL_TRUE;
        }



    public:
        /*! \brief Creates an object of type CALModel3D, sets its records and returns it as a pointer; it defines the cellular automaton structure.
        */
        CALModelUnsafe(std::array<COORDINATE_TYPE, DIMENSION>& _coordinates, //!< Dimensions  of cellular space.
                       NEIGHBORHOOD_pointer _calNeighborhood, //!< Class that identifies the type of neighbourhood relation to be used.
                       enum opencal::calCommon::CALSpaceBoundaryCondition _CAL_TOROIDALITY, //!< Enumerator that identifies the type of cellular space: toroidal or non-toroidal.
                       enum opencal::calCommon::CALOptimization _CAL_OPTIMIZATION //!< Enumerator used for specifying the active cells optimization or no optimization.
                       ): opencal::CALModel<DIMENSION,NEIGHBORHOOD, COORDINATE_TYPE>(_coordinates, _calNeighborhood, _CAL_TOROIDALITY, _CAL_OPTIMIZATION)
        {

        }

        /*! \brief Creates and adds a new substate to CALModel::pQ_arrays and return a pointer to it.
    */
        template <class PAYLOAD, calCommon::SUBSTATE_OPT OPT= calCommon::NO_OPT>
        CALSubstateUnsafe<PAYLOAD, DIMENSION, COORDINATE_TYPE, OPT>* addSubstateUnsafe(){
            using SUBSTATE = CALSubstateUnsafe<PAYLOAD, DIMENSION , COORDINATE_TYPE, OPT>;
            using SUBSTATE_pointer = CALSubstateUnsafe<PAYLOAD, DIMENSION , COORDINATE_TYPE,OPT>*;

            CALSubstateWrapper<DIMENSION , COORDINATE_TYPE>** pQ_array_tmp = this->pQ_arrays;
            CALSubstateWrapper<DIMENSION , COORDINATE_TYPE>** pQ_array_new;
            SUBSTATE_pointer Q;
            uint i;


            pQ_array_new = new CALSubstateWrapper<DIMENSION , COORDINATE_TYPE>* [this->sizeof_pQ_arrays + 1];


            for (i = 0; i < this->sizeof_pQ_arrays; i++)
            {
                pQ_array_new[i] = pQ_array_tmp[i];
            }

            if (!allocSubstateUnsafe<PAYLOAD, OPT>(Q))
            {
                return NULL;
            }

            pQ_array_new[this->sizeof_pQ_arrays] = (CALSubstateWrapper<DIMENSION , COORDINATE_TYPE>*) Q;

            this->pQ_arrays = pQ_array_new;


            delete [] pQ_array_tmp;

            this->sizeof_pQ_arrays++;
            return Q;
        }

        /*! \brief Creates a new single-layer byte substate and returns a pointer to it.
            Note that sinlgle-layer substates are not added to CALModel3D::pQ*_array because
            they do not nedd to be updated.
        */
        template <class PAYLOAD, calCommon::SUBSTATE_OPT OPT= calCommon::NO_OPT>
        CALSubstateUnsafe<PAYLOAD, DIMENSION, COORDINATE_TYPE, OPT>* addSingleLayerSubstateUnsafe(){
            using SUBSTATE_type = CALSubstateUnsafe<PAYLOAD, DIMENSION , COORDINATE_TYPE,OPT>;
            using SUBSTATE_pointer = CALSubstateUnsafe<PAYLOAD, DIMENSION , COORDINATE_TYPE,OPT>*;
            using BUFFER_pointer = CALBuffer<PAYLOAD, DIMENSION, COORDINATE_TYPE>*;
            using BUFFER_type = CALBuffer<PAYLOAD, DIMENSION, COORDINATE_TYPE>;

            SUBSTATE_pointer Q;
            BUFFER_pointer current = new BUFFER_type (this->coordinates);
            Q = new SUBSTATE_type(current, NULL);
            return Q;

        }

        /*! \brief Adds a transition function's elementary process to the CALModel3D::elementary_processes array of callbacks pointers.
            Note that the function calGlobalTransitionFunction3D calls a substates' update after each elementary process.
        */

        /*! \brief Sets the n-th neighbor of the cell (i,j) of the matrix flags to
            CAL_TRUE and increments the couter sizeof_active_flags.
        */
        void addActiveCellX(std::array<COORDINATE_TYPE, DIMENSION>& indexes,	//!< Row coordinate of the central cell.
                            int n	//!< Index of the n-th neighbor to be added.
                            )
        {
            int linearIndex = calCommon::cellLinearIndex<DIMENSION,COORDINATE_TYPE>(indexes, this->coordinates);
            int neighboorIndex =  CALNeighborPool<DIMENSION,COORDINATE_TYPE>::getNeighborN(linearIndex,n);
            this->activeCells->setFlag(neighboorIndex, CAL_TRUE);
        }

        /*! \brief Sets the n-th neighbor of the cell (i,j) of the matrix flags to
            CAL_TRUE and increments the couter sizeof_active_flags.
        */
        void addActiveCellX(int linearIndex,	//!< Row coordinate of the central cell.
                            int n	//!< Index of the n-th neighbor to be added.
                            )
        {
            int neighboorIndex =  CALNeighborPool<DIMENSION,COORDINATE_TYPE>::getNeighborN(linearIndex,n);
            this->activeCells->setFlag(neighboorIndex, CAL_TRUE);
        }


    };

}



#endif

