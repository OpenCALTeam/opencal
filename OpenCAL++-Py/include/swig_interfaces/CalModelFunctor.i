%module(directors="1") calModelFunctor
%{
	#include "CalModelFunctor.h"
%}

%include "CalModelFunctor.h"

%template(StopConditionFunction3D) CalModelFunctor<CALModel3D,CALbyte>;

