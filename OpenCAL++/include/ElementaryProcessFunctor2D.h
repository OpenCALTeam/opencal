
#ifndef ELEMENTARYPROCESS2D_H_
#define ELEMENTARYPROCESS2D_H_


class ElementaryProcessFunctor2D {
public:

								ElementaryProcessFunctor2D();
								virtual void run(struct CALModel2D* model, int i, int j) =0;
								virtual void operator()(CALModel2D* model, int i, int j);
								virtual ~ElementaryProcessFunctor2D();
};


#endif /* EMENTARYPROCESS2D_H_ */
