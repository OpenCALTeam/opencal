#ifndef _OPENCAL_IMAGE_PROCESSING_
#define _OPENCAL_IMAGE_PROCESSING_

#include<type_traits>
#include<tuple>
#include<OpenCAL++/calModel.h>
#include<tuple>
#include<vector>
#include<cmath>
#include<OpenCAL++/functional_utilities.h>
#include<functional>

  template<class T>
  void save (const T *array, const std::string pathOutput,int rows, int cols, int type)
  {


      cv::Mat mat (rows, cols, type);
      int linearIndex =0;
      //printf ("%d %d \n", mat.rows, mat.cols);
      for (int i = 0; i < mat.rows; ++i) {
          for (int j = 0; j < mat.cols; ++j, ++linearIndex)
              mat.at<T>(i,j) = array[linearIndex];
      }
      cv::imwrite(pathOutput, mat);
      return;
  }



  template<class T>
  T *loadImage(int size, const std::string& path){
  //    printf("sto qui\n");
      cv::Mat mat= cv::imread(path);

      //int size = mat.rows * mat.cols;
      T* vec = new T [size];
      int linearIndex = 0;

      for (int i = 0; i < mat.rows; ++i) {
          for (int j = 0; j < mat.cols; ++j, ++linearIndex) {
              T& bgra = mat.at<T>(i, j);
              vec[linearIndex] = bgra;
          }
      }


      return vec;
  }


  template<typename T>
  class CALLBACKTYPE{
  public:
      typedef std::function<void(const T*, const std::string&)> SAVECALLBACK;
      typedef std::function<T*(int size, const std::string&)>    LOADCALLBACK;

  };


namespace opencal {

  template< uint DIMENSION , template <uint, class...> class _NEIGHBORHOOD, class ...TYPES>
  class Kernel {

    public:
     typedef std::tuple<TYPES...> PAYLOAD;
     typedef std::vector<PAYLOAD> VEC_TYPE;

     Kernel(uint size) : data(size) {};
     Kernel(const VEC_TYPE& _data) : data(_data) {};

    PAYLOAD& operator[](const int& idx)
      {
       return data[idx];
      };



    protected:
     VEC_TYPE data ;

  };


  template< uint DIMENSION , template <uint , class...> class _NEIGHBORHOOD , class FLOATING>
  class UniformKernel : public Kernel<DIMENSION, _NEIGHBORHOOD, FLOATING> {

    typedef Kernel<DIMENSION , _NEIGHBORHOOD,  FLOATING> SUPER;
    typedef _NEIGHBORHOOD<DIMENSION> NEIGHBORHOOD;
    typedef typename NEIGHBORHOOD::COORDINATE_TYPE COORDINATE_TYPE;

    public:
     UniformKernel(uint size) : SUPER(size) {};
     UniformKernel(const typename SUPER::VEC_TYPE& _data) = delete ;

    protected:
    std::array<double,DIMENSION> sigma;
    std::array<double,DIMENSION> mu;

     void initKernel(){
       const auto&  indices = NEIGHBORHOOD::getNeighborhoodIndices();
       const auto size = indices.size();
       for(int i =0 ; i < size ; ++i ){
          this->data[i] = 1/size;
       }

     }

  };


  template< uint DIMENSION , template <uint , class...> class _NEIGHBORHOOD , class FLOATING>
  class GaussianKernel : public Kernel<DIMENSION, _NEIGHBORHOOD, FLOATING> {

    typedef Kernel<DIMENSION , _NEIGHBORHOOD,  FLOATING> SUPER;
    typedef _NEIGHBORHOOD<DIMENSION> NEIGHBORHOOD;
    typedef typename NEIGHBORHOOD::COORDINATE_TYPE COORDINATE_TYPE;
    //static auto phi = [] (const double sigma, const
    public:
     GaussianKernel(uint size,std::array<double,DIMENSION> _sigma, std::array<double,DIMENSION> _mu) : SUPER(size) , sigma(_sigma) , mu(_mu) {initKernel();};
     GaussianKernel(const typename SUPER::VEC_TYPE& _data) = delete ;

    protected:
    std::array<double,DIMENSION> sigma;
    std::array<double,DIMENSION> mu;

     void initKernel(){
       const auto&  indices = NEIGHBORHOOD::getNeighborhoodIndices();
       for(int i =0 ; i < indices.size() ; ++i ){
          this->data[i] = getGaussianVal(indices[i]);
       }

     }

     double getGaussianVal(const std::array<COORDINATE_TYPE,DIMENSION>& x){


       double prodSig2 = fold(sigma.begin(),sigma.end(),1,
           []( double acc,  double s) -> double {return s*s*acc; }
           );
       double coeff = 1/(2*M_PIl * prodSig2);

       double exponent = 0;
       for(int i = 0 ; i < DIMENSION ; ++i){
          double p = (x[i] - mu[i])/sigma[i];
          p*=2;
          exponent+=p;
       }
       exponent*=0.5;

       return coeff * exp(exponent);
     }

  };



  template< uint DIMENSION, template <uint, class...> class _NEIGHBORHOOD , template<uint , class...> class _KERNEL , template<uint, class...> class _SUBSTATE, class COORDINATE_TYPE>
  class ConvolutionFilter: public opencal::CALLocalFunction<DIMENSION, _NEIGHBORHOOD<DIMENSION> ,  COORDINATE_TYPE>{

  typedef _KERNEL<DIMENSION> KERNEL;
  typedef _SUBSTATE<DIMENSION> SUBSTATE;

  typedef CALModel<DIMENSION, _NEIGHBORHOOD<DIMENSION>, COORDINATE_TYPE> *MODEL_pointer;


    protected:
      KERNEL* kernel;
      SUBSTATE* substate;

      //some substate may need convolution filter to be applied differenlty.
      //Consider to overload this function to obtain the desired result
      virtual void applyConvolution(MODEL_pointer model, std::array<COORDINATE_TYPE,DIMENSION>& indices, KERNEL* kernel){
       typename SUBSTATE::PAYLOAD newVal;

       for(int i=0 ; i < model->getNeighborhoodSize() ; ++i)
          newVal += (*kernel)[i] * substate->getX(indices,i);

       substate->setElement(indices,newVal);

      }


    public:

      ConvolutionFilter(const SUBSTATE* _sub , const KERNEL* k) : substate(_sub) , kernel(k) {};
      ConvolutionFilter() = delete;



  inline  void run(MODEL_pointer model, std::array<COORDINATE_TYPE,DIMENSION>& indices){
        applyConvolution(model, indices, kernel);
      }



  };




} //namespace opencal


#endif //_OPENCAL_IMAGE_PROCESSING_


