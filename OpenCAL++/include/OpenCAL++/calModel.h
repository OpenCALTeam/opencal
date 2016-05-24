// (C) Copyright University of Calabria and others.
// All rights reserved. This program and the accompanying materials
// are made available under the terms of the GNU Lesser General Public License
// (LGPL) version 2.1 which accompanies this distribution, and is available at
// http://www.gnu.org/licenses/lgpl-2.1.html
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
// Lesser General Public License for more details.

#ifndef CALMODEL_H_
#define CALMODEL_H_

#include<memory>
#include<array>

#include <OpenCAL++/calCommon.h>
#include <OpenCAL++/calBuffer.h>
#include<OpenCAL++/calActiveCells.h>
#include<OpenCAL++/calSubstate.h>
#include <OpenCAL++/calIndexesPool.h>
#include <OpenCAL++/calNeighborPool.h>
#include <OpenCAL++/calElementaryProcessFunctor.h>
#include<vector>




namespace opencal {


        template<uint DIMENSION, class NEIGHBORHOOD, typename COORDINATE_TYPE = uint>
        class CALModel {

            typedef CALModel CALMODEL_type;
            typedef CALModel* CALMODEL_pointer;
            typedef  std::array<COORDINATE_TYPE, DIMENSION>& CALCELL_INDIXES_constreference;



            typedef NEIGHBORHOOD* NEIGHBORHOOD_pointer;
            typedef NEIGHBORHOOD& NEIGHBORHOOD_reference;

            typedef CALFunction<DIMENSION , NEIGHBORHOOD , COORDINATE_TYPE> CALCallbackFunc_type;
            //typedef void(*CALCallbackFunc_type)(CALMODEL_pointer, CALCELL_INDIXES_constreference);
            typedef CALCallbackFunc_type& CALCallbackFunc_reference;
            typedef CALCallbackFunc_type* CALCallbackFunc_pointer;
            typedef CALCallbackFunc_type** CALCallbackFunc_pointer_pointer;

            using CACallback = CALCallbackFunc_pointer;



        public:

            //typedefs here

            /******************************************************************************
                            DEFINITIONS OF FUNCTIONS PROTOTYPES
            *******************************************************************************/
            /*! \brief Constructor of the object CALModel, sets and inizializes its records; it defines the cellular automaton object.
           */
            CALModel(std::array<COORDINATE_TYPE, DIMENSION>& _coordinates, //!< Dimensions  of cellular space.
                     NEIGHBORHOOD_pointer _calNeighborhood, //!< Class that identifies the type of neighbourhood relation to be used.
                     enum opencal::calCommon::CALSpaceBoundaryCondition _CAL_TOROIDALITY, //!< Enumerator that identifies the type of cellular space: toroidal or non-toroidal.
                     enum opencal::calCommon::CALOptimization _CAL_OPTIMIZATION //!< Enumerator used for specifying the active cells optimization or no optimization.
            ): coordinates(_coordinates) , CAL_TOROIDALITY(_CAL_TOROIDALITY) , CAL_OPTIMIZATION(_CAL_OPTIMIZATION)
            {

                this->size = opencal::calCommon::multiplier<DIMENSION,uint>(coordinates,0);

                //initialize indexpool and neighbor pool
                CALNeighborPool<DIMENSION, COORDINATE_TYPE> :: init(coordinates, CAL_TOROIDALITY);
                CALIndexesPool<DIMENSION, COORDINATE_TYPE>:: init(coordinates,size);



                this->sizeof_X = 0;

                if (this->CAL_OPTIMIZATION == calCommon::CAL_OPT_ACTIVE_CELLS) {
                    CALBuffer<bool , DIMENSION , COORDINATE_TYPE>* flags = new CALBuffer<bool , DIMENSION , COORDINATE_TYPE> (this->coordinates);
                    flags->setBuffer(false);
                    this->activeCells = new opencal::CALActiveCells< DIMENSION , COORDINATE_TYPE>(flags, 0);
                }
                else
                    this->activeCells = NULL;


                this->pQ_arrays = NULL;
                this->sizeof_pQ_arrays = 0;


                this->X_id = _calNeighborhood;

                if (X_id != NULL){

                    //NEIGHBORHOOD::defineNeighborhood();

                    this->addNeighbors(this->X_id->getNeighborhoodIndices());
                }





            }

            ~CALModel (){
                for (int i = 0; i < this->sizeof_pQ_arrays; ++i) {
                    delete pQ_arrays[i];
                }
                delete [] pQ_arrays;

                CALIndexesPool<DIMENSION , COORDINATE_TYPE>::destroy();
                CALNeighborPool<DIMENSION , COORDINATE_TYPE>::destroy();
                delete activeCells;
//                delete X_id;



            }

            /*! \brief Sets a certain cell of the matrix flags to true and increments the
            couter sizeof_active_flags.
        */
            void addActiveCell(std::array<COORDINATE_TYPE, DIMENSION>& indexes){
                this->activeCells->setElementFlags(indexes, this->coordinates, CAL_TRUE);
            }
            void addActiveCell(int linearIndex){
                this->activeCells->setFlag(linearIndex, CAL_TRUE);
            }

            /*! \brief Sets a specific cell of the matrix flags to false and decrements the
            couter sizeof_active_flags.
        */
            void removeActiveCell(std::array<COORDINATE_TYPE, DIMENSION>& indexes){
                this->activeCells->setElementFlags(indexes, this->coordinates,  CAL_FALSE);
            }
            void removeActiveCell(int linearIndex){
                this->activeCells->setFlag(linearIndex, CAL_FALSE);
            }


            /*! \brief Perform the update of CALActiveCells object.
        */
            void updateActiveCells(){
                activeCells->update();
            }

            /*! \brief Adds a neighbour to CALNeighbourPool.
            */
            void  addNeighbor(auto indexes){
                CALNeighborPool<DIMENSION , COORDINATE_TYPE>::addNeighbor(indexes);
                this->sizeof_X++;
            }

            /*! \brief Adds a neighbours to CALNeighbourPool.
            */
            void  addNeighbors(int** indexes, size_t size){
                int n = 0;
                for (n = 0; n < size; n++){
                    CALNeighborPool<DIMENSION, COORDINATE_TYPE>::addNeighbor(indexes[n]);
                    this->sizeof_X ++;
                }
            }

            /*! \brief Adds a neighbours to CALNeighbourPool.
            */
            void  addNeighbors(const auto& indices){
                for (auto el: indices){
                    CALNeighborPool<DIMENSION, COORDINATE_TYPE>::addNeighbor(el);
                    this->sizeof_X ++;
                }
            }

            void addElementaryProcess(CACallback _elementary_process)
            {
              elementary_processes.push_back(_elementary_process);
            }

//            void applyElementaryProcess(CACallback _elementary_process) //!< Pointer to a transition function's elementary process
//            {
//                int i, n;

//                if (this->activeCells) //Computationally active cells optimization.
//                {
//                    int sizeCurrent = this->activeCells->getSizeCurrent();
//                    for (n=0; n<sizeCurrent; n++)
//                        (*_elementary_process)(this,calCommon::cellMultidimensionalIndices<DIMENSION,COORDINATE_TYPE>(this->activeCells->getCells()[n]));
//                }
//                else //Standart cicle of the transition function
//                {

//                    for (i=0; i<this->size; i++){
//                    auto indices = calCommon:: cellMultidimensionalIndices<DIMENSION,COORDINATE_TYPE>(i);
//                        (*_elementary_process)(this,indices);
//                    }

//                }
//            }


            void globalTransitionFunction()
            {
                //The global transition function.
                //It applies transition function elementary processes sequentially.
                //Note that a substates' update is performed after each elementary process.
                for (const auto elementaty_process : elementary_processes){

                  //applying the b-th elementary process
//                    this->applyElementaryProcess(elementaty_process);
                    //updating substates
//                    this-> update();
                    (*elementaty_process)(this);
                }
            }

            void update()
            {
                //updating active cells
                if (this->CAL_OPTIMIZATION == calCommon :: CAL_OPT_ACTIVE_CELLS)
                    this->updateActiveCells();


                for (int i = 0; i < this->sizeof_pQ_arrays; ++i)
                {
                    pQ_arrays[i]->update(this->activeCells);
                }


            }
            /*! \brief Creates and adds a new substate to CALModel::pQ_arrays and return a pointer to it.
        */
            template <class PAYLOAD, calCommon::SUBSTATE_OPT OPT= calCommon::NO_OPT>
            CALSubstate<PAYLOAD, DIMENSION, COORDINATE_TYPE, OPT>* addSubstate(){
                using SUBSTATE = CALSubstate<PAYLOAD, DIMENSION , COORDINATE_TYPE, OPT>;
                using SUBSTATE_pointer = CALSubstate<PAYLOAD, DIMENSION , COORDINATE_TYPE,OPT>*;

                CALSubstateWrapper<DIMENSION , COORDINATE_TYPE>** pQ_array_tmp = this->pQ_arrays;
                CALSubstateWrapper<DIMENSION , COORDINATE_TYPE>** pQ_array_new;
                SUBSTATE_pointer Q;
                uint i;


                pQ_array_new = new CALSubstateWrapper<DIMENSION , COORDINATE_TYPE>* [this->sizeof_pQ_arrays + 1];


                for (i = 0; i < this->sizeof_pQ_arrays; i++)
                {
                    pQ_array_new[i] = pQ_array_tmp[i];
                }

                if (!allocSubstate<PAYLOAD, OPT>(Q))
                {
                    return NULL;
                }

                pQ_array_new[this->sizeof_pQ_arrays] = (CALSubstateWrapper<DIMENSION , COORDINATE_TYPE>*) Q;

                this->pQ_arrays = pQ_array_new;


                delete [] pQ_array_tmp;

                this->sizeof_pQ_arrays++;
                return Q;
            }

            /*! \brief Creates a new single-layer substate and returns a pointer to it.
                Note that sinlgle-layer substates are not added to CALModel::pQ_arrays because
                they do not need to be updated.
            */
            template <class PAYLOAD, calCommon::SUBSTATE_OPT OPT= calCommon::NO_OPT>
            CALSubstate<PAYLOAD, DIMENSION, COORDINATE_TYPE, OPT>* addSingleLayerSubstate(){
                using SUBSTATE_type = CALSubstate<PAYLOAD, DIMENSION , COORDINATE_TYPE,OPT>;
                using SUBSTATE_pointer = CALSubstate<PAYLOAD, DIMENSION , COORDINATE_TYPE,OPT>*;
                using BUFFER_pointer = CALBuffer<PAYLOAD, DIMENSION, COORDINATE_TYPE>*;
                using BUFFER_type = CALBuffer<PAYLOAD, DIMENSION, COORDINATE_TYPE>;

                SUBSTATE_pointer Q;
                BUFFER_pointer current = new BUFFER_type (this->coordinates);
                Q = new SUBSTATE_type(current, NULL);

            }


            template <class PAYLOAD, calCommon::SUBSTATE_OPT OPT= calCommon::NO_OPT>
            void initSubstate(CALSubstate<PAYLOAD, DIMENSION, COORDINATE_TYPE,OPT>*& Q, PAYLOAD value) {
                if (this->activeCells)
                {
                    Q->setActiveCellsBufferCurrent(this->activeCells, value);
                    if(Q->getNext())
                        Q->setActiveCellsBufferNext(this->activeCells, value);

                }
                else
                {
                    Q->setCurrentBuffer(value);
                    if(Q->getNext())
                        Q->setNextBuffer(value);

                }
            }

            template <class PAYLOAD, calCommon::SUBSTATE_OPT OPT= calCommon::NO_OPT>
            void initSubstateNext(CALSubstate<PAYLOAD, DIMENSION, COORDINATE_TYPE, OPT>*& Q, PAYLOAD value) {
                if (this->activeCells)
                    Q->setActiveCellsBuffer(this->activeCells, value);
                else
                    Q->setNextBuffer(value);
            }

            template <class PAYLOAD, calCommon::SUBSTATE_OPT OPT= calCommon::NO_OPT>
            void init(CALSubstate<PAYLOAD, DIMENSION, COORDINATE_TYPE,OPT>*& Q, std::array<COORDINATE_TYPE, DIMENSION>& indexes, PAYLOAD value) {

                int linearIndex = calCommon::cellLinearIndex<DIMENSION, COORDINATE_TYPE>(indexes, this->coordinates);
                (*Q->getCurrent())[linearIndex] = value;
                (*Q->getNext())[linearIndex] = value;
            }






            //getters and setters
            uint getDimension() {
                return DIMENSION;
            }

            int getSize ()
            {
                return this->size;
            }

            std::array<COORDINATE_TYPE, DIMENSION>& getCoordinates ()
            {
                return this->coordinates;
            }


            int getNeighborhoodSize ()
            {
                return this->sizeof_X;
            }

            auto getActiveCells ()
            {
                return activeCells;
            }


            void setNeighborhoodSize (int sizeof_X)
            {
                this->sizeof_X = sizeof_X;
            }

//            opencal::CALActiveCells<DIMENSION,COORDINATE_TYPE>* getActiveCells ()
//            {
//                return activeCells;
//            }


            template <class PAYLOAD>
            CALSubstate<PAYLOAD, DIMENSION, COORDINATE_TYPE>* getSubstate(const int idx){
                return static_cast<CALSubstate<PAYLOAD, DIMENSION, COORDINATE_TYPE>*>(pQ_arrays[idx]);
            }



        private:
            enum opencal::calCommon::CALSpaceBoundaryCondition CAL_TOROIDALITY;    //!< Type of cellular space: toroidal or non-toroidal.


            std::array <COORDINATE_TYPE, DIMENSION> coordinates;
            uint size;

            int sizeof_X;                //!< Number of cells belonging to the neighbourhood. Note that predefined neighbourhoods include the central cell.
            NEIGHBORHOOD_pointer X_id;    //!< Class that define the Neighbourhood relation.


            opencal::CALActiveCells<DIMENSION,COORDINATE_TYPE>* activeCells;			//!< Computational Active cells object. if activecells==NULL no optimization is applied.
            enum opencal::calCommon::CALOptimization CAL_OPTIMIZATION;    //!< Type of optimization used. It can be CAL_NO_OPT or CAL_OPT_ACTIVE_CELLS.


            CALSubstateWrapper<DIMENSION , COORDINATE_TYPE> ** pQ_arrays; //!< Substates array.
            int sizeof_pQ_arrays;           //!< Number of substates.

            //CALCallbackFunc_pointer_pointer elementary_processes; //!< Array of transition function's elementary processes callback functions. Note that a substates' update must be performed after each elementary process has been applied to each cell of the cellular space.
            //int num_of_elementary_processes; //!< Number of function pointers to the transition functions's elementary processes callbacks.

            std::vector<CACallback> elementary_processes;

            //protected methods
        protected:

            template<class PAYLOAD, calCommon::SUBSTATE_OPT OPT= calCommon::NO_OPT>
            calCommon::CALbyte allocSubstate(CALSubstate<PAYLOAD , DIMENSION , COORDINATE_TYPE, OPT>*& Q){
                using BUFFER = CALBuffer<PAYLOAD, DIMENSION, COORDINATE_TYPE>;
                using SUBSTATE = CALSubstate<PAYLOAD, DIMENSION , COORDINATE_TYPE, OPT>;

                BUFFER* current = new BUFFER(this->coordinates);
                BUFFER* next = new BUFFER(this->coordinates);
                Q = new SUBSTATE (current, next, this->coordinates);

                if (!Q->getCurrent() || !Q->getNext()){
                    return CAL_FALSE;
                }

                return CAL_TRUE;
            }

        };


    } //namespace opencal
#endif //CALMODEL_H_
