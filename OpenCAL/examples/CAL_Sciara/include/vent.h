#ifndef vent_h
#define vent_h

#include <stdio.h>
#include <vector>
using std::vector;

#define EMISSION_RATE_FILE_ERROR     0
#define EMISSION_RATE_FILE_OK        1

//---------------------------------------------------------------------------
class TEmissionRate {
public:
	int vent_id() {
		return _vent_id;
	}
	void set_vent_id(int id) {
		_vent_id = id;
	}
	double& operator[](int i) {
		return _emission_rate[i];
	}
	vector<double>& emission_rate() {
		return _emission_rate;
	}
	unsigned int size() {
		return _emission_rate.size();
	}
	void print() {
		printf("vent_id = %d\nemission_rate\n", _vent_id);
		for (unsigned int i = 0; i < _emission_rate.size(); i++)
			printf("%f\n", _emission_rate[i]);
	}
private:
	int _vent_id;
	vector<double> _emission_rate;
};
//---------------------------------------------------------------------------
class TVent {
public:
	int x() {
		return _x;
	}
	int y() {
		return _y;
	}
	int vent_id() {
		return _vent_id;
	}
	void set_x(int x) {
		_x = x;
	}
	void set_y(int y) {
		_y = y;
	}
	void set_vent_id(int id) {
		_vent_id = id;
	}
	bool setEmissionRate(vector<TEmissionRate> er_vec, int id) {
		bool found = false;
		for (unsigned int i = 0; i < er_vec.size(); i++)
			if (er_vec[i].vent_id() == id) {
				found = true;
				_emission_rate = er_vec[i].emission_rate();
			}
		return found;
	}
	double& operator[](int i) {
		return _emission_rate[i];
	}
	unsigned int size() {
		return _emission_rate.size();
	}
	double thickness(double sim_elapsed_time, double Pt, unsigned int emission_time, double Pac) {
		unsigned int i;

		i = (unsigned int) (sim_elapsed_time / emission_time);
		if (i >= _emission_rate.size())
			return 0;
		else
			return _emission_rate[i] / Pac * Pt;
	}
	void print() {
		printf("vent_id = %d\nemission_rate\n", _vent_id);
		for (unsigned int i = 0; i < _emission_rate.size(); i++)
			printf("%f\n", _emission_rate[i]);
	}
private:
	int _x;
	int _y;
	int _vent_id;
	vector<double> _emission_rate;
};
//---------------------------------------------------------------------------
/*extern unsigned int emission_time;
 extern vector<TEmissionRate> emission_rate;
 extern vector<TVent> vent;*/

void initVents(int* Mv, int lx, int ly, vector<TVent>& vent);
void addVent(int x, int y, int vent_id, vector<TVent>& vent);
void removeLastVent(vector<TVent>& vent);
int loadEmissionRates(FILE *f, unsigned int& emission_time, vector<TEmissionRate>& er_vec, vector<TVent> &vent);
int loadOneEmissionRates(FILE *f, unsigned int vent_id, vector<TEmissionRate>& er_vec);
int defineVents(const vector<TEmissionRate>& emission_rate, vector<TVent>& vent);

void rebuildVentsMatrix(int* Mv, int lx, int ly, vector<TVent>& vent);
void saveEmissionRates(FILE *f, unsigned int emission_time, vector<TEmissionRate>& er_vec);
void printEmissionRates(unsigned int emission_time, vector<TEmissionRate>& er_vec);
//---------------------------------------------------------------------------

#endif
