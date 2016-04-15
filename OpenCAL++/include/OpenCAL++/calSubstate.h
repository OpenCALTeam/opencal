//
// Created by knotman on 13/04/16.
//

#ifndef OPENCAL_ALL_CALSUBSTATE_H_H
#define OPENCAL_ALL_CALSUBSTATE_H_H


#include <OpenCAL++/calBuffer.h>
//#include<OpenCAL++/calNeighborPool.h>
#include <OpenCAL++/calActiveCells.h>
#include <OpenCAL++/calConverterIO.h>

namespace opencal {

    template<uint DIMENSION, typename COORDINATE_TYPE = uint>
    class CALSubstateWrapper {

    public:
        typedef CALConverterIO<DIMENSION , COORDINATE_TYPE> CALCONVERTERIO_type;
        typedef CALConverterIO<DIMENSION , COORDINATE_TYPE>& CALCONVERTERIO_reference;
        typedef CALConverterIO<DIMENSION , COORDINATE_TYPE>* CALCONVERTERIO_pointer;

        virtual ~ CALSubstateWrapper() { }

        virtual void update(opencal::CALActiveCells<DIMENSION, COORDINATE_TYPE> *activeCells) = 0;

        virtual void saveSubstate(std::array<COORDINATE_TYPE, DIMENSION>& _coordinates, CALCONVERTERIO_pointer calConverterInputOutput,
                                  char *path) = 0;

        virtual void loadSubstate(std::array<COORDINATE_TYPE, DIMENSION>& coordinates, CALCONVERTERIO_pointer calConverterInputOutput,
                                  char *path) = 0;
    };

    template<class PAYLOAD, uint DIMENSION, typename COORDINATE_TYPE = uint>
    class CALSubstate : public CALSubstateWrapper<DIMENSION , COORDINATE_TYPE> {

        typedef CALBuffer <PAYLOAD, DIMENSION, COORDINATE_TYPE> BUFFER_TYPE;
        typedef CALBuffer <PAYLOAD, DIMENSION, COORDINATE_TYPE>& BUFFER_TYPE_REF;
        typedef CALBuffer <PAYLOAD, DIMENSION, COORDINATE_TYPE>* BUFFER_TYPE_PTR;
        typedef CALBuffer <PAYLOAD, DIMENSION, COORDINATE_TYPE>* BUFFER_TYPE_REF_TO_PTR;


        typedef CALActiveCells<DIMENSION , COORDINATE_TYPE> CALACTIVECELLS_type;
        typedef CALActiveCells<DIMENSION , COORDINATE_TYPE>& CALACTIVECELLS_reference;
        typedef CALActiveCells<DIMENSION , COORDINATE_TYPE>* CALACTIVECELLS_pointer;

        typedef CALConverterIO<DIMENSION , COORDINATE_TYPE>* CALCONVERTERIO_pointer;


    private:
        BUFFER_TYPE_PTR current;    //!< Current linearised matrix of the substate, used for reading purposes.
        BUFFER_TYPE_PTR next;        //!< Next linearised matrix of the substate, used for writing purposes.

    public:


        CALSubstate(){
            this->current   = nullptr;
            this->next      = nullptr;
        }

        virtual ~ CALSubstate(){
            delete this->current;
            delete this->next;
        }

        CALSubstate(BUFFER_TYPE_PTR _current, BUFFER_TYPE_PTR _next){
            this->current   = _current;
            this->next      = _next;
        }

        CALSubstate(const CALSubstate& obj){
            this->current   = obj.current;
            this->next      = obj.next;
        }

        BUFFER_TYPE_REF_TO_PTR getCurrent(){
            return this->current;
        }

        BUFFER_TYPE_REF_TO_PTR getNext(){
            return this->next;
        }

        void setCurrent(BUFFER_TYPE_PTR _current){
            if (this->current)
                delete this->current;
            this->current = current;
        }

        void setNext(BUFFER_TYPE_PTR _next){
            if (this->next)
                delete this->next;
            this->next = next;
        }

        void setActiveCellsBufferCurrent(CALACTIVECELLS_pointer activeCells, PAYLOAD value){
            this->current->setActiveCellsBuffer(activeCells->getCells(), activeCells->getSizeCurrent(), value);
        }

        void setActiveCellsBufferNext(CALACTIVECELLS_pointer activeCells, PAYLOAD value){
            this->next->setActiveCellsBuffer(activeCells->getCells(), activeCells->getSizeCurrent(), value);
        }

        /*! \brief Copies the next 3D buffer of a byte substate to the current one: current = next.
                If the active cells optimization is considered, it only updates the active cells.
            */
        virtual void update(CALACTIVECELLS_pointer activeCells){
            if (activeCells)
                this->current->copyActiveCellsBuffer(next, activeCells->getCells(), activeCells->getSizeCurrent());
            else
            {
                *current = *next;
            }
        }

        virtual void saveSubstate(std::array<COORDINATE_TYPE, DIMENSION>& _coordinates, CALCONVERTERIO_pointer calConverterInputOutput,
                                  char *path);

        virtual void loadSubstate(std::array<COORDINATE_TYPE, DIMENSION>& _coordinates, CALCONVERTERIO_pointer calConverterInputOutput,
                                  char *path);

        void setElementCurrent(int *indexes, std::array<COORDINATE_TYPE, DIMENSION>& _coordinates, PAYLOAD value);

        void setElement(int *indexes, std::array<COORDINATE_TYPE, DIMENSION>& _coordinates, PAYLOAD value);

        PAYLOAD getElement(int *indexes, std::array<COORDINATE_TYPE, DIMENSION>& _coordinates);

        PAYLOAD getElementNext(int *indexes, std::array<COORDINATE_TYPE, DIMENSION>& _coordinates);

        PAYLOAD getX(int linearIndex, int n){

        }

        PAYLOAD getX(int *indexes, std::array<COORDINATE_TYPE, DIMENSION>& _coordinates, int n){

        }

        PAYLOAD getElement(int linearIndex);

        PAYLOAD getElementNext(int linearIndex);

        void setElementCurrent(int linearIndex, PAYLOAD value);

        void setElement(int linearIndex, PAYLOAD value);

        void setCurrentBuffer(PAYLOAD value);

        void setNextBuffer(PAYLOAD value);

        CALSubstate& operator=(const CALSubstate &b);


    };





} //namespace opencal

#endif //OPENCAL_ALL_CALSUBSTATE_H_H
