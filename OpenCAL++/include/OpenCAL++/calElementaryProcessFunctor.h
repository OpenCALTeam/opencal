//
// Created by knotman on 14/04/16.
//

#ifndef OPENCAL_ALL_CALELEMENTARYPROCESSFUNCTOR_H
#define OPENCAL_ALL_CALELEMENTARYPROCESSFUNCTOR_H

/*! \brief Class that defines transition function's elementary processes.
*/

template<uint DIMENSION, class NEIGHBORHOOD, typename COORDINATE_TYPE = int>
class CALModel;

template<uint DIMENSION, class NEIGHBORHOOD, typename COORDINATE_TYPE = int>
class CALElementaryProcessFunctor {
    typedef CALModel<DIMENSION , NEIGHBORHOOD , COORDINATE_TYPE> CALMODEL_type;
    typedef CALModel<DIMENSION , NEIGHBORHOOD , COORDINATE_TYPE>* CALMODEL_pointer;
public:

    CALElementaryProcessFunctor();

    /*! \brief Method that has to ridefined in concrete derived class in order to specify the necessary steps for elementary process.
    */
    virtual void run(CALMODEL_pointer calModel, int* indexes) =0;
    virtual void operator()(CALMODEL_pointer calModel, int* indexes);
    virtual ~CALElementaryProcessFunctor();
};


#endif //OPENCAL_ALL_CALELEMENTARYPROCESSFUNCTOR_H
