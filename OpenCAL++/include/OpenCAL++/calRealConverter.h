#ifndef OPENCAL_ALL_CALREALCONVERTERIO_H
#define OPENCAL_ALL_CALREALCONVERTERIO_H


#include <OpenCAL++/calCommon.h>

#include <string>
#include <sstream>
#include <iomanip>
#pragma once

namespace opencal {

    // string -> signed integral
    inline int stoi(const std::string& s){
       return std::stoi(s);
    }

    inline long int stol(const std::string& s){
       return std::stol(s);
    }

    inline long long int stoll(const std::string& s){
       return std::stoll(s);
    }

    // string -> unsigned integral
    inline unsigned long int stoul(const std::string& s){
       return std::stol(s);
    }

    inline unsigned long long int stoull(const std::string& s){
       return std::stoll(s);
    }

    // string -> floating
    inline float stof(const std::string& s){
       return std::stof(s);
    }

    inline double stod(const std::string& s){
       return std::stol(s);
    }

    inline long double stold(const std::string& s){
       return std::stoll(s);
    }

    //string to boolean (aka int short)
    inline bool stob(const std::string& s){
       return (bool)std::stoi(s);
    }

template<class T>
std::string tostring_fn_(const T& s,const int n = 6) {
 	std::ostringstream out;
	out<<std::setprecision(n) <<s;
	return out.str();
}



template<class T>
const auto tostring_fn(const int n = 6) {
    using namespace std::placeholders;
 	auto f = std::bind(tostring_fn_<T>, _1,n);
    return f;
}

//interface for all singleton converters
template<typename T>
class Converter{

public:
    static auto& getInstance() ;
    template<class STR_TYPE>
    static T deserialize(STR_TYPE input){

    }
    template<class STR_TYPE>
    static STR_TYPE serialize(T& output);


    Converter(Converter const&) = delete;
    void operator=(Converter const&)  = delete;
    Converter() = default;

    template<class STR_TYPE>
    std::string operator()(T& p){return serialize<STR_TYPE>(p);};
    template<class STR_TYPE>
    T operator()(STR_TYPE& s){return deserialize<STR_TYPE>(s);};

};

class CALRealConverter : public Converter<double>{
    using Converter::Converter;
protected:

public:
static CALRealConverter& getInstance()
        {
            static CALRealConverter   instance; // Guaranteed to be destroyed.
                                                // Instantiated on first use.
            return instance;
        }

    static double deserialize(std::string input)
    {
        return (std::stof(input));
    }


    static std::string serialize(double output)
    {
       std::string converted = std::to_string(output);
       return converted;

    }
};

class CALIntConverter : public Converter<int> {
public:

    template<class STR_TYPE>
    static int deserialize(STR_TYPE input)
    {
        return (std::stoi(input));
    }

template<class STR_TYPE>
    static STR_TYPE serialize(int output)
    {
       std::string converted = std::to_string(output);
       return converted;

    }
};


}//namespace opencal
#endif
