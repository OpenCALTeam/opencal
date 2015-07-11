#include<ElementaryProcessFunctor2D.h>

ElementaryProcessFunctor2D::ElementaryProcessFunctor2D() {

}

ElementaryProcessFunctor2D::~ElementaryProcessFunctor2D() {

}
 void ElementaryProcessFunctor2D::operator()(CALModel2D* model, int i, int j)
{
	this->run(model,i,j);
}

void ElementaryProcessFunctor2D::run(CALModel2D* model, int i, int j)
{

}
