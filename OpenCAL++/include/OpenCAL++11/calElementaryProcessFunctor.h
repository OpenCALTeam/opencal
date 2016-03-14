#ifndef calElementaryProcess_h_
#define calElementaryProcess_h_

class CALModel;

/*! \brief Class that defines transition function's elementary processes.
*/
class CALElementaryProcessFunctor {
public:

    CALElementaryProcessFunctor();

    /*! \brief Method that has to ridefined in concrete derived class in order to specify the necessary steps for elementary process.
    */
    virtual void run(CALModel* calModel, int* indexes) =0;
    virtual void operator()(CALModel* calModel, int* indexes);
    virtual ~CALElementaryProcessFunctor();
};


#endif /* EMENTARYPROCESS_H_ */
