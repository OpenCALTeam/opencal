/*
 * Particle.h
 *
 *  Created on: Jan 7, 2016
 *      Author: knotman
 */

#ifndef PARTICLE_H_
#define PARTICLE_H_

#include<cmath>
#include<array>

template <unsigned int DIMENSION , typename T = double>

class Particle {
public:

	uint id,group,type;
	T volume;
	T mass;
	//position
  std::array<DIMENSION , T> p;
	//velocity
  std::array<DIMENSION , T> v;



	Particle(){};
	virtual ~Particle(){}

	void print(std::ostream& os , const std::string sep = " , "){

    os
		<<id<<sep
		<<group<<sep
		<<type<<sep
		<<volume<<sep
		<<mass<<sep
		<<p.x<<sep
		<<v.x;
	}
};

#endif /* PARTICLE_H_ */
