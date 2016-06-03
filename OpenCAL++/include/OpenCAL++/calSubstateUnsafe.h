 
#ifndef calSubstateUnsafe_h
#define calSubstateUnsafe_h


#include<OpenCAL++/calSubstate.h>


namespace opencal {
    template<class PAYLOAD, uint DIMENSION, typename COORDINATE_TYPE = uint, calCommon::SUBSTATE_OPT OPT= calCommon::NO_OPT>
    class CALSubstateUnsafe :public CALSubstate<PAYLOAD, DIMENSION, COORDINATE_TYPE, OPT>
    {
         typedef CALBuffer <PAYLOAD, DIMENSION, COORDINATE_TYPE>* BUFFER_TYPE_PTR;
    public:

        CALSubstateUnsafe (BUFFER_TYPE_PTR _current, BUFFER_TYPE_PTR _next, std::array<COORDINATE_TYPE,DIMENSION> _coordinates):
            CALSubstate<PAYLOAD, DIMENSION, COORDINATE_TYPE, OPT>(_current, _next, _coordinates)
        {

        }

        CALSubstateUnsafe(std::array<COORDINATE_TYPE,DIMENSION> _coordinates): CALSubstate<PAYLOAD, DIMENSION, COORDINATE_TYPE, OPT>(_coordinates)
        {

        }

        /*! \brief Inits the cell (i, j) n-th neighbour of a byte substate to value;
        it updates both the current and next matrix at the position (i, j).
        This operation is unsafe since it writes value in a neighbor, both on the current and next matrix.
    */
        void initX(std::array<COORDINATE_TYPE,DIMENSION>& indexes, int n, PAYLOAD value)
        {
             int linearIndex = calCommon::cellLinearIndex<DIMENSION,COORDINATE_TYPE>(indexes, this->coordinates);
            int neighboorIndex =  CALNeighborPool<DIMENSION,COORDINATE_TYPE>::getNeighborN(linearIndex,n);

            (*this->current)[neighboorIndex] = value;
            (*this->next)[neighboorIndex] = value;
        }

        /*! \brief Inits the cell (i, j) n-th neighbour of a byte substate to value;
        it updates both the current and next matrix at the position (i, j).
        This operation is unsafe since it writes value in a neighbor, both on the current and next matrix.
    */

        void initX(int linearIndex, int n, PAYLOAD value)
        {
            int neighboorIndex =  CALNeighborPool<DIMENSION,COORDINATE_TYPE>::getNeighborN(linearIndex,n);

            (*this->current)[neighboorIndex] = value;
            (*this->next)[neighboorIndex] = value;
        }

        /*! \brief Returns the cell (i, j) value of a byte substate from the next matrix.
        This operation is unsafe since it reads a value from the next matrix.
    */

        PAYLOAD getNextElement(std::array<COORDINATE_TYPE,DIMENSION>& indexes)
        {
             int linearIndex = calCommon::cellLinearIndex<DIMENSION,COORDINATE_TYPE>(indexes, this->coordinates);
            return (*this->next)[linearIndex];
        }

        /*! \brief Returns the cell (i, j) value of a byte substate from the next matrix.
        This operation is unsafe since it reads a value from the next matrix.
    */

        PAYLOAD getNextElement(int linearIndex)
        {
            return (*this->next)[linearIndex];
        }

        /*! \brief Returns the cell (i, j) n-th neighbor value of a byte substate from the next matrix.
        This operation is unsafe since it reads a value from the next matrix.
    */

        PAYLOAD getNextX(std::array<COORDINATE_TYPE,DIMENSION>& indexes, int n)
        {
             int linearIndex = calCommon::cellLinearIndex<DIMENSION,COORDINATE_TYPE>(indexes, this->coordinates);
            int neighboorIndex =  CALNeighborPool<DIMENSION,COORDINATE_TYPE>::getNeighborN(linearIndex,n);

            return (*this->next)[neighboorIndex];
        }

        /*! \brief Returns the cell (i, j) n-th neighbor value of a byte substate from the next matrix.
        This operation is unsafe since it reads a value from the next matrix.
    */

        PAYLOAD getNextX(int linearIndex, int n)
        {
            int neighboorIndex =  CALNeighborPool<DIMENSION,COORDINATE_TYPE>::getNeighborN(linearIndex,n);

            return (*this->next)[neighboorIndex];
        }

        /*! \brief Sets the value of the n-th neighbor of the cell (i, j) of a byte substate.
        This operation is unsafe since it writes a value in a neighbor of the next matrix.
    */

        void setX(std::array<COORDINATE_TYPE,DIMENSION>& indexes, int n, PAYLOAD value)
        {
             int linearIndex = calCommon::cellLinearIndex<DIMENSION,COORDINATE_TYPE>(indexes, this->coordinates);
            int neighboorIndex =  CALNeighborPool<DIMENSION,COORDINATE_TYPE>::getNeighborN(linearIndex,n);
            (*this->next)[neighboorIndex] = value;
        }

        /*! \brief Sets the value of the n-th neighbor of the cell (i, j) of a byte substate.
        This operation is unsafe since it writes a value in a neighbor of the next matrix.
    */

        void setX(int linearIndex, int n, PAYLOAD value)
        {

            int neighboorIndex =  CALNeighborPool<DIMENSION,COORDINATE_TYPE>::getNeighborN(linearIndex,n);
            (*this->next)[neighboorIndex] = value;

        }

        /*! \brief Sets the value of the n-th neighbor of the cell (i, j)x of a byte substate of the CURRENT matri.
        This operation is unsafe since it writes a value directly to the current matrix.
    */

        void setCurrentX(std::array<COORDINATE_TYPE,DIMENSION>& indexes, int n,	PAYLOAD value )
        {
             int linearIndex = calCommon::cellLinearIndex<DIMENSION,COORDINATE_TYPE>(indexes, this->coordinates);
            int neighboorIndex =  CALNeighborPool<DIMENSION,COORDINATE_TYPE>::getNeighborN(linearIndex,n);

            (*this->current)[neighboorIndex] = value;
        }



        /*! \brief Sets the value of the n-th neighbor of the cell (i, j)x of a byte substate of the CURRENT matri.
        This operation is unsafe since it writes a value directly to the current matrix.
    */

        void setCurrentX(int linearIndex, int n, PAYLOAD value)
        {
            int neighboorIndex =  CALNeighborPool<DIMENSION,COORDINATE_TYPE>::getNeighborN(linearIndex,n);

            (*this->current)[neighboorIndex] = value;
        }
    };

}


#endif
