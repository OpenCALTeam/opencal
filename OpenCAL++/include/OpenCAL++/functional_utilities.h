//
// Created by knotman on 12/04/16.
//

#ifndef OPENCAL_ALL_FUNCTIONAL_UTILITIES_H
#define OPENCAL_ALL_FUNCTIONAL_UTILITIES_H


namespace opencal{

    template< typename  Iterator ,  typename Lambda >
    void for_each(Iterator s, Iterator e, Lambda l){
        while(s != e){
            l(*s);
            s++;
        }
    }

    //Lambda has type: D -> T -> D
    template<typename D ,
            typename Iterator,
            typename Lambda>
    D fold( Iterator s, Iterator e, const D &a, Lambda l){
        D acc = a ;

        while(s != e){
            acc = l(acc,*s);
            s++;
        }
        return acc;
    }


}//namespace opencal


#endif //OPENCAL_ALL_FUNCTIONAL_UTILITIES_H
