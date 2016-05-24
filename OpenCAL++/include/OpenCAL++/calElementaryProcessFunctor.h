
#ifndef CALELEMENTARYPROCESSFUNCTOR_H_
#define CALELEMENTARYPROCESSFUNCTOR_H_


/*! \brief Class that defines transition function's elementary processes.
*/
namespace opencal {


template<uint DIMENSION, class NEIGHBORHOOD, typename COORDINATE_TYPE>
class CALModel;


template<uint DIMENSION, class NEIGHBORHOOD, typename COORDINATE_TYPE = uint>
class CALFunction {
protected:
    typedef CALModel<DIMENSION, NEIGHBORHOOD, COORDINATE_TYPE> CALMODEL_type;
    typedef CALModel<DIMENSION, NEIGHBORHOOD, COORDINATE_TYPE> *CALMODEL_pointer;

public:

    CALFunction()
    {

    }

    /*! \brief Method that has to ridefined in concrete derived class in order to specify the necessary steps for elementary process.
        */
    //    virtual void run(CALMODEL_pointer calModel, std::array<COORDINATE_TYPE,DIMENSION>& indexes) = 0;

    virtual void operator()(CALMODEL_pointer calModel)
    {
        execute(calModel);
    }

    virtual void execute(CALMODEL_pointer calModel) = 0;

    virtual ~CALFunction()
    {

    }
};

template<uint DIMENSION, class NEIGHBORHOOD, typename COORDINATE_TYPE = uint>
class CALGlobalFunction: public CALFunction<DIMENSION, NEIGHBORHOOD, COORDINATE_TYPE> {
protected:
    typedef CALModel<DIMENSION, NEIGHBORHOOD, COORDINATE_TYPE> CALMODEL_type;
    typedef CALModel<DIMENSION, NEIGHBORHOOD, COORDINATE_TYPE> *CALMODEL_pointer;

    enum calCommon :: CALUpdateMode UPDATE_MODE;	//!< Callbacks substates' update mode; it can be CAL_UPDATE_EXPLICIT or CAL_UPDATE_IMPLICIT.
public:

    CALGlobalFunction(enum calCommon :: CALUpdateMode _UPDATE_MODE): UPDATE_MODE(_UPDATE_MODE)
    {

    }

    /*! \brief Method that has to ridefined in concrete derived class in order to specify the necessary steps for elementary process.
        */
    virtual void run(CALMODEL_pointer calModel) = 0;

    virtual void operator()(CALMODEL_pointer calModel)
    {
        execute(calModel);
    }

    virtual void execute(CALMODEL_pointer calModel)
    {

        this->run(calModel);
        if (UPDATE_MODE == calCommon :: CAL_UPDATE_IMPLICIT)
            calModel->update();
    }

    virtual ~CALGlobalFunction()
    {

    }
};

template<uint DIMENSION, class NEIGHBORHOOD, typename COORDINATE_TYPE = uint>
class CALLocalFunction: public CALFunction<DIMENSION, NEIGHBORHOOD, COORDINATE_TYPE> {
protected:
    typedef CALModel<DIMENSION, NEIGHBORHOOD, COORDINATE_TYPE> CALMODEL_type;
    typedef CALModel<DIMENSION, NEIGHBORHOOD, COORDINATE_TYPE> *CALMODEL_pointer;

public:

    CALLocalFunction()
    {

    }

    /*! \brief Method that has to ridefined in concrete derived class in order to specify the necessary steps for elementary process.
        */
    virtual void run(CALMODEL_pointer calModel, std::array<COORDINATE_TYPE,DIMENSION>& indexes) = 0;

    virtual void operator()(CALMODEL_pointer calModel)
    {
        execute(calModel);
    }

    virtual void execute(CALMODEL_pointer calModel)
    {
        int i, n;

        opencal::CALActiveCells<DIMENSION,COORDINATE_TYPE>* activeCells = calModel->getActiveCells();
        if (activeCells) //Computationally active cells optimization.
        {
            int sizeCurrent = activeCells->getSizeCurrent();
            for (n=0; n<sizeCurrent; n++)
                this->run(calModel,calCommon::cellMultidimensionalIndices<DIMENSION,COORDINATE_TYPE>(activeCells->getCells()[n]));
        }
        else //Standart cicle of the transition function
        {

            int size = calModel->getSize();
            for (i=0; i<size; i++){
                auto indices = calCommon:: cellMultidimensionalIndices<DIMENSION,COORDINATE_TYPE>(i);
                this->run(calModel,indices);
            }

        }

        calModel->update();
    }

    virtual ~CALLocalFunction()
    {

    }
};


}

#endif
