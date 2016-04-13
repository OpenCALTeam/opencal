//
// Created by knotman on 12/04/16.
//

#ifndef OPENCAL_ALL_CALACTIVECELLS_H
#define OPENCAL_ALL_CALACTIVECELLS_H


#include <OpenCAL++/calCommon.h>
#include <OpenCAL++/calBuffer.h>

namespace opencal {
/*! \brief Active cells class.
*/

    template<unsigned int DIMENSION,  typename COORDINATE_TYPE = uint>
    class CALActiveCells {

        typedef CALBuffer<bool, DIMENSION,COORDINATE_TYPE> BUFFER_type;
        typedef CALBuffer<bool, DIMENSION,COORDINATE_TYPE>* BUFFER_pointer;

    private:
        BUFFER_pointer flags;            //!< Buffer of flags having the substates' dimension: flag is true if the corresponding cell is active, false otherwise.
        int *cells;                            //!< Array of computational active cells.
        int size_current;                    //!< Number of active cells in the current step.
        int size_next;                      //!< Number of true flags.

    public:
        /*! \brief CALActiveCells' construsergctor with no parameter.
        */
        CALActiveCells(){
            this->size_current = 0;
            this->size_next = 0;
            this->cells = nullptr;
            this->flags = nullptr;
        }

        /*! \brief CALActiveCells' constructor.
        */
        CALActiveCells(BUFFER_pointer flags, int size_next){
            this->flags = flags;
            this->size_current = 0;

            this->size_next = size_next;
            this->cells = nullptr;
        }

        CALActiveCells(BUFFER_pointer flags, int size_next, int *cells, int size_current){
            this->flags = flags;
            this->size_next= size_next;
            this->cells = cells;
            this->size_current = size_current;
        }

        /*! \brief CALActiveCells' destructor.
        */
        ~ CALActiveCells(){
            delete [] this->cells;
            delete this->flags;
        }

        CALActiveCells(const CALActiveCells &obj){
            this->flags = obj.flags;
            this->size_current = obj.size_current;
            cells = new int [obj.size_next];

            for (int i = 0; i < obj.size_next; ++i)
            {
                cells[i] = obj.cells[i];
            }
            this->size_next = obj.size_next;
        }

        /*! \brief CALActiveCells' setter and getter methods.
        */
        BUFFER_pointer getFlags(){
            return flags;
        }

        void setFlags(BUFFER_pointer _flags){
            this->flags = _flags;
        }

        int getSizeNext(){
            return this->size_next;
        }

        void setSizeNext(int _size_next){
            this->size_next = _size_next;
        }

        int *getCells(){
            return this->cells;
        }

        void setCells(int* _cells){
            this-> cells = _cells;
        }

        int getSizeCurrent(){
            return this->size_current;
        }

        void setSizeCurrent(int _size_current){
            this->size_current = _size_current;
        }

        bool getElementFlags(int *indexes, int *coordinates, int dimension){
            return this->flags->getElement(indexes, coordinates,dimension);
        }

        /*! \brief Sets the cell of coordinates indexes of the matrix flags to parameter value. It
         * increments the couter size_current if value is true, it decreases otherwise.
        */
        void setElementFlags(int *indexes, int *coordinates, int dimension, bool value){
            if (value && !flags->getElement(indexes, coordinates,dimension))
            {
                flags->setElement(indexes, coordinates,dimension, value);
                size_next++;
            }
            else if (!value && flags->getElement(indexes, coordinates,dimension))
            {
                flags->setElement(indexes, coordinates,dimension, value);
                size_next--;
            }
        }

        /*! \brief Puts the cells marked as actives in flags into the array of active cells
            cells and sets its dimension, to size_of_actives, i.e. the actual
            number of active cells.
        */
        void update(){
            int i, n;
            if(this->cells)
            {
                delete [] this->cells;
            }

            this->size_current = this->size_next;
            if (size_current == 0)
                return;

            this->cells = new int [this->size_current];

            n = 0;
            int flagSize = flags->getSize();
            for(i = 0; i < flagSize; i++)
            {
                if ((*flags)[i])
                {
                    cells[n] = i;
                    n++;
                }
            }
        }

        bool getFlag(int linearIndex){
            return (*flags)[linearIndex];
        }

        /*! \brief Sets the cell [linearIndex] of the matrix flags to parameter value. It
         * increments the couter size_current if value is true, it decreases otherwise.
        */
        void setFlag(int linearIndex, bool value){
            (*flags)[linearIndex] = value;
            size_next--;
        }


    };

} //namespace opencal
#endif //OPENCAL_ALL_CALACTIVECELLS_H
