#ifndef calModelFunctor_h
#define calModelFunctor_h

template<class MODEL,class RET_TYPE>
class CALModelFunctor{
public:
	virtual RET_TYPE run(MODEL*)=0;
    CALModelFunctor(){}
	virtual RET_TYPE operator()(MODEL* model){
		return this->run(model);
	}
    virtual ~CALModelFunctor(){}

};




#endif
