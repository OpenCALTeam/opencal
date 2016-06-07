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

  template<class Tuple, std::size_t N>
struct TuplePrinter{

  static void print(const Tuple& t){
    TuplePrinter<Tuple, N-1>::print(t);
    std::cout<<", "<<std::get<N>(t);
  }
};

template<class Tuple>
struct TuplePrinter<Tuple,1>{

  static void print(const Tuple& t){
    std::cout<<std::get<0>(t);
  }
};

template<class... Args>
void print_tuple(const std::tuple<Args...>& t){
  std::cout<<"(";
  TuplePrinter<decltype(t),sizeof...(Args)>::print(t);
  std::cout<<")";
}

template< uint _DIMENSION ,  class _NEIGHBORHOOD, class ...TYPES>
  class Kernel {

    public:
     typedef std::tuple<TYPES...> PAYLOAD;
     typedef std::vector<PAYLOAD> VEC_TYPE;

     template<typename = typename std::enable_if<_DIMENSION == _NEIGHBORHOOD::_DIMENSION>::type>
      Kernel() : data(_NEIGHBORHOOD::getNeighborhoodIndices().size()) {};

     Kernel(uint size) : data(size) {};
     Kernel(const VEC_TYPE& _data) : data(_data) {};

    PAYLOAD& operator[](const int& idx)
      {
       return data[idx];
      };

   virtual  void initKernel() = 0;

    void print(){
      using namespace std;
      for(int i=0 ; i < data.size(); ++i){
        cout<<_NEIGHBORHOOD::getNeighborhoodIndices()[i][0]<<" , "<<_NEIGHBORHOOD::getNeighborhoodIndices()[i][1]<<" ";
        print_tuple(data[i]);
        cout<<endl;
      }
    }


    protected:
     VEC_TYPE data ;

  };


  template< uint DIMENSION , class _NEIGHBORHOOD , class FLOATING>
  class UniformKernel : public Kernel<DIMENSION, _NEIGHBORHOOD, FLOATING> {

    typedef Kernel<DIMENSION , _NEIGHBORHOOD,  FLOATING> SUPER;
    typedef _NEIGHBORHOOD NEIGHBORHOOD;
    typedef typename NEIGHBORHOOD::COORDINATE_TYPE COORDINATE_TYPE;

    public:
      UniformKernel() : SUPER() {initKernel();};
     UniformKernel(uint size) : SUPER(size) {initKernel();};

     UniformKernel(const typename SUPER::VEC_TYPE& _data) = delete ;



    protected:
    std::array<double,DIMENSION> sigma;
    std::array<double,DIMENSION> mu;

     void initKernel(){
       const auto&  indices = NEIGHBORHOOD::getNeighborhoodIndices();
       const auto size = indices.size();
       for(int i =0 ; i < size ; ++i ){
         std::get<0>(this->data[i]) = static_cast<FLOATING>(1/static_cast<FLOATING>(size));
       }

     }

  };


  template< uint DIMENSION ,class _NEIGHBORHOOD , class FLOATING>
  class GaussianKernel : public Kernel<DIMENSION, _NEIGHBORHOOD, FLOATING> {

    typedef Kernel<DIMENSION , _NEIGHBORHOOD,  FLOATING> SUPER;
    typedef _NEIGHBORHOOD NEIGHBORHOOD;
    typedef typename NEIGHBORHOOD::COORDINATE_TYPE COORDINATE_TYPE;
    //static auto phi = [] (const double sigma, const
    public:

    GaussianKernel(std::array<double,DIMENSION> _sigma, std::array<double,DIMENSION> _mu) : SUPER() , sigma(_sigma) , mu(_mu) {initKernel();};

     GaussianKernel(uint size,std::array<double,DIMENSION> _sigma, std::array<double,DIMENSION> _mu) : SUPER(size) , sigma(_sigma) , mu(_mu) {initKernel();};
     GaussianKernel(const typename SUPER::VEC_TYPE& _data) = delete ;

    protected:
    std::array<double,DIMENSION> sigma;
    std::array<double,DIMENSION> mu;

     void initKernel(){
       const auto&  indices = NEIGHBORHOOD::getNeighborhoodIndices();
       FLOATING sum = 0;
       for(int i =0 ; i < indices.size() ; ++i ){
         FLOATING val = getGaussianVal(indices[i]);
         sum+=val;
         std::get<0>(this->data[i]) = val;
       }
       //normalize the kernel
       for(int i =0 ; i < this->data.size() ; ++i )
          std::get<0>(this->data[i])/=sum;
     }

     double getGaussianVal(const std::array<COORDINATE_TYPE,DIMENSION>& x){


       double prodSigma = fold(sigma.begin(),sigma.end(),1.0,
           []( double acc,  double s) -> double {return s*acc; }
           );
       double coeff = 1/(2*M_PIl * prodSigma);

       double exponent = 0;
       for(int i = 0 ; i < DIMENSION ; ++i){
          exponent+= ( (x[i] - mu[i]) * (x[i] -mu[i]) ) /(2*(sigma[i]* sigma[i]));
       }

       return coeff * exp(-exponent);
     }

  };



  template< uint DIMENSION,  class _NEIGHBORHOOD , class _KERNEL , class _SUBSTATE, class COORDINATE_TYPE=uint>
  class ConvolutionFilter: public opencal::CALLocalFunction<DIMENSION, _NEIGHBORHOOD ,  COORDINATE_TYPE>{
    public:
  typedef _KERNEL KERNEL;
  typedef _SUBSTATE SUBSTATE;

  typedef CALModel<DIMENSION, _NEIGHBORHOOD, COORDINATE_TYPE> *MODEL_pointer;


    protected:
      KERNEL* kernel;
      SUBSTATE* substate;

      //some substate may need convolution filter to be applied differenlty.
      //Consider to overload this function to obtain the desired result
     virtual void applyConvolution(MODEL_pointer model, std::array<COORDINATE_TYPE,DIMENSION>& indices, KERNEL* kernel) =0;

    public:

      ConvolutionFilter(SUBSTATE* _sub , KERNEL* k) : substate(_sub) , kernel(k) {};
      ConvolutionFilter() = delete;



  inline  void run(MODEL_pointer model, std::array<COORDINATE_TYPE,DIMENSION>& indices){
        applyConvolution(model, indices, kernel);
      }



  };




} //namespace opencal


#endif //_OPENCAL_IMAGE_PROCESSING_


