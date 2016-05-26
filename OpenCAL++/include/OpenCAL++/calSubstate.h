//
// Created by knotman on 13/04/16.
//

#ifndef OPENCAL_ALL_CALSUBSTATE_H_H
#define OPENCAL_ALL_CALSUBSTATE_H_H


#include <OpenCAL++/calBuffer.h>
#include<OpenCAL++/calNeighborPool.h>
#include <OpenCAL++/calActiveCells.h>
#include <OpenCAL++/calManagerIO.h>

namespace opencal {

template<uint DIMENSION, typename COORDINATE_TYPE = uint>
class CALSubstateWrapper {

public:
    //        typedef CALConverterIO<DIMENSION , COORDINATE_TYPE> CALCONVERTERIO_type;
    //        typedef CALConverterIO<DIMENSION , COORDINATE_TYPE>& CALCONVERTERIO_reference;
    //        typedef CALConverterIO<DIMENSION , COORDINATE_TYPE>* CALCONVERTERIO_pointer;

    virtual ~ CALSubstateWrapper() { }

    virtual void update(opencal::CALActiveCells<DIMENSION, COORDINATE_TYPE> *activeCells) = 0;

    //        virtual void raveSubstate(CALCONVERTERIO_pointer calConverterInputOutput, char *path) = 0;

    //        virtual void loadSubstate(CALCONVERTERIO_pointer calConverterInputOutput, char *path) = 0;
};

template<class PAYLOAD, uint DIMENSION, typename COORDINATE_TYPE = uint, calCommon::SUBSTATE_OPT OPT= calCommon::NO_OPT>
class CALSubstate : public CALSubstateWrapper<DIMENSION , COORDINATE_TYPE> {

    typedef CALBuffer <PAYLOAD, DIMENSION, COORDINATE_TYPE> BUFFER_TYPE;
    typedef CALBuffer <PAYLOAD, DIMENSION, COORDINATE_TYPE>& BUFFER_TYPE_REF;
    typedef CALBuffer <PAYLOAD, DIMENSION, COORDINATE_TYPE>* BUFFER_TYPE_PTR;
    typedef CALBuffer <PAYLOAD, DIMENSION, COORDINATE_TYPE>* BUFFER_TYPE_REF_TO_PTR;


    typedef CALActiveCells<DIMENSION , COORDINATE_TYPE> CALACTIVECELLS_type;
    typedef CALActiveCells<DIMENSION , COORDINATE_TYPE>& CALACTIVECELLS_reference;
    typedef CALActiveCells<DIMENSION , COORDINATE_TYPE>* CALACTIVECELLS_pointer;

    typedef CALManagerIO<DIMENSION , COORDINATE_TYPE>* CALCONVERTERIO_pointer;


private:
    BUFFER_TYPE_PTR current;    //!< Current linearised matrix of the substate, used for reading purposes.
    BUFFER_TYPE_PTR next;        //!< Next linearised matrix of the substate, used for writing purposes.
    std::array<COORDINATE_TYPE,DIMENSION> coordinates;

//    std::vector <int> modified;
//    bool* flagModified;
public:


    CALSubstate(std::array<COORDINATE_TYPE,DIMENSION> _coordinates){
        this->current   = nullptr;
        this->next      = nullptr;
        this->coordinates = _coordinates;
    }

    virtual ~ CALSubstate(){
        delete this->current;
        delete this->next;
    }

    CALSubstate(BUFFER_TYPE_PTR _current, BUFFER_TYPE_PTR _next, std::array<COORDINATE_TYPE,DIMENSION> _coordinates){
        this->current   = _current;
        this->next      = _next;
        this->coordinates = _coordinates;
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

    template <class CALCONVERTER, class STR_TYPE = std::string>
    void saveSubstate(CALCONVERTER& calConverterInputOutput, const STR_TYPE& path)
    {
        this->current-> template saveBuffer(this->coordinates,calConverterInputOutput, path);
    }


    template <class CALCONVERTER, class STR_TYPE = std::string>
    void loadSubstate(CALCONVERTER& calConverterInputOutput, const STR_TYPE& path)
    {

        delete current;
        this->current = new BUFFER_TYPE (this->coordinates, path, calConverterInputOutput);
        if (this->next)
            *next = *current;
    }

    void setElementCurrent(std::array<COORDINATE_TYPE,DIMENSION>& indexes, PAYLOAD value)
    {
        this->current->setElement(indexes, this->coordinates, value);
    }

    void setElement(std::array<COORDINATE_TYPE,DIMENSION>& indexes, PAYLOAD value)
    {
        this->next->setElement(indexes,coordinates, value);
    }

    PAYLOAD getElement(std::array<COORDINATE_TYPE,DIMENSION>& indexes)
    {
        return this->current->getElement(indexes, this->coordinates);
    }

    PAYLOAD getElementNext(std::array<COORDINATE_TYPE,DIMENSION>& indexes)
    {
        return this->next->getElement(indexes, this->coordinates);
    }

    PAYLOAD getX(int linearIndex, int n) const
    {
        return (*this->current)[CALNeighborPool<DIMENSION,COORDINATE_TYPE>::getNeighborN(linearIndex,n)];
    }

    PAYLOAD getX(std::array<COORDINATE_TYPE,DIMENSION>& indexes, int n){

        int linearIndex = calCommon::cellLinearIndex<DIMENSION,COORDINATE_TYPE>(indexes, this->coordinates);
        return (*this->current)[CALNeighborPool<DIMENSION,COORDINATE_TYPE>::getNeighborN(linearIndex,n)];
    }

    PAYLOAD getElement(int linearIndex) const
    {
        return (*this->current)[linearIndex];
    }

    PAYLOAD getElementNext(int linearIndex) const
    {
        return (*this->next)[linearIndex];
    }

    void setElementCurrent(int linearIndex, PAYLOAD value)
    {
        (*this->current)[linearIndex] = value;
    }

    void setElement(int linearIndex, PAYLOAD value)
    {
        (*this->next)[linearIndex] = value;
    }

    void setCurrentBuffer(PAYLOAD value)
    {
        this->current->setBuffer(value);
    }

    void setNextBuffer(PAYLOAD value)
    {
        this->next->setBuffer(value);
    }

    CALSubstate& operator=(const CALSubstate &b)
    {
        if (this != &b)
        {

            //TODO SISTEMARE
            BUFFER_TYPE_PTR currentTmp = new BUFFER_TYPE ();
            BUFFER_TYPE_PTR nextTmp = new BUFFER_TYPE ();

            *currentTmp = *b.current;
            *nextTmp = *b.next;

            //        delete current;
            //        delete next;

            this->current = currentTmp;
            this->next = nextTmp;
        }
        return *this;

    }



};


template<class PAYLOAD, uint DIMENSION, typename COORDINATE_TYPE>
class CALSubstate<PAYLOAD,DIMENSION,COORDINATE_TYPE,calCommon::OPT> : public CALSubstateWrapper<DIMENSION , COORDINATE_TYPE> {

    typedef CALBuffer <PAYLOAD, DIMENSION, COORDINATE_TYPE> BUFFER_TYPE;
    typedef CALBuffer <PAYLOAD, DIMENSION, COORDINATE_TYPE>& BUFFER_TYPE_REF;
    typedef CALBuffer <PAYLOAD, DIMENSION, COORDINATE_TYPE>* BUFFER_TYPE_PTR;
    typedef CALBuffer <PAYLOAD, DIMENSION, COORDINATE_TYPE>* BUFFER_TYPE_REF_TO_PTR;


    typedef CALActiveCells<DIMENSION , COORDINATE_TYPE> CALACTIVECELLS_type;
    typedef CALActiveCells<DIMENSION , COORDINATE_TYPE>& CALACTIVECELLS_reference;
    typedef CALActiveCells<DIMENSION , COORDINATE_TYPE>* CALACTIVECELLS_pointer;

    typedef CALManagerIO<DIMENSION , COORDINATE_TYPE>* CALCONVERTERIO_pointer;


private:
    BUFFER_TYPE_PTR current;    //!< Current linearised matrix of the substate, used for reading purposes.
    BUFFER_TYPE_PTR next;        //!< Next linearised matrix of the substate, used for writing purposes.
    std::array<COORDINATE_TYPE,DIMENSION> coordinates;

    std::vector <int> modified;
    bool* flagModified;
public:


    CALSubstate(std::array<COORDINATE_TYPE,DIMENSION> _coordinates){
        this->current   = nullptr;
        this->next      = nullptr;
        this->coordinates = _coordinates;
        initialize();
    }

    virtual ~ CALSubstate(){
        delete this->current;
        delete this->next;
        delete this->flagModified;
    }

    CALSubstate(BUFFER_TYPE_PTR _current, BUFFER_TYPE_PTR _next, std::array<COORDINATE_TYPE,DIMENSION> _coordinates){
        this->current   = _current;
        this->next      = _next;
        this->coordinates = _coordinates;
        initialize();
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
            for(int i= 0; i<modified.size(); i++)
            {
                (*current)[modified[i]] = (*next)[modified[i]];
                flagModified[modified[i]] = false;
            }
            modified.clear();
            //                            *current = *next;
        }
    }

    template <class CALCONVERTER, class STR_TYPE = std::string>
    void saveSubstate(CALCONVERTER& calConverterInputOutput, const STR_TYPE& path)
    {
        this->current-> template saveBuffer(this->coordinates,calConverterInputOutput, path);
    }


    template <class CALCONVERTER, class STR_TYPE = std::string>
    void loadSubstate(CALCONVERTER& calConverterInputOutput, const STR_TYPE& path)
    {

        delete current;
        this->current = new BUFFER_TYPE (this->coordinates, path, calConverterInputOutput);
        if (this->next)
            *next = *current;
    }

    void setElementCurrent(std::array<COORDINATE_TYPE,DIMENSION>& indexes, PAYLOAD value)
    {
        this->current->setElement(indexes, this->coordinates, value);
    }

    void setElement(std::array<COORDINATE_TYPE,DIMENSION>& indexes, PAYLOAD value)
    {
        uint linearIndex = opencal::calCommon::cellLinearIndex<DIMENSION,COORDINATE_TYPE>(indexes,coordinates);

        addModifiedCell(linearIndex, value);
        this->next->setElement(linearIndex, value);
    }

    PAYLOAD getElement(std::array<COORDINATE_TYPE,DIMENSION>& indexes)
    {
        return this->current->getElement(indexes, this->coordinates);
    }

    PAYLOAD getElementNext(std::array<COORDINATE_TYPE,DIMENSION>& indexes)
    {
        return this->next->getElement(indexes, this->coordinates);
    }

    PAYLOAD getX(int linearIndex, int n) const
    {
        return (*this->current)[CALNeighborPool<DIMENSION,COORDINATE_TYPE>::getNeighborN(linearIndex,n)];
    }

    PAYLOAD getX(std::array<COORDINATE_TYPE,DIMENSION>& indexes, int n){

        int linearIndex = calCommon::cellLinearIndex<DIMENSION,COORDINATE_TYPE>(indexes, this->coordinates);
        return (*this->current)[CALNeighborPool<DIMENSION,COORDINATE_TYPE>::getNeighborN(linearIndex,n)];
    }

    PAYLOAD getElement(int linearIndex) const
    {
        return (*this->current)[linearIndex];
    }

    PAYLOAD getElementNext(int linearIndex) const
    {
        return (*this->next)[linearIndex];
    }

    void setElementCurrent(int linearIndex, PAYLOAD value)
    {
        (*this->current)[linearIndex] = value;
    }

    void setElement(int linearIndex, PAYLOAD value)
    {
        addModifiedCell(linearIndex, value);
        (*this->next)[linearIndex] = value;
    }

    void setCurrentBuffer(PAYLOAD value)
    {
        this->current->setBuffer(value);
    }

    void setNextBuffer(PAYLOAD value)
    {
        this->next->setBuffer(value);
    }

    CALSubstate& operator=(const CALSubstate &b)
    {
        if (this != &b)
        {

            //TODO SISTEMARE
            BUFFER_TYPE_PTR currentTmp = new BUFFER_TYPE ();
            BUFFER_TYPE_PTR nextTmp = new BUFFER_TYPE ();

            *currentTmp = *b.current;
            *nextTmp = *b.next;

            //        delete current;
            //        delete next;

            this->current = currentTmp;
            this->next = nextTmp;
        }
        return *this;

    }

private:
    void initialize()
    {
        int size = calCommon::multiplier <DIMENSION,COORDINATE_TYPE>(coordinates,0);
        this->flagModified = new bool[size] {false};


    }
    void addModifiedCell(int linearIndex, PAYLOAD value)
    {
        if (flagModified[linearIndex] == false && (*next)[linearIndex] != value)
        {
            modified.push_back(linearIndex);

            flagModified[linearIndex] = true;

        }
    }


};




} //namespace opencal

#endif //OPENCAL_ALL_CALSUBSTATE_H_H
