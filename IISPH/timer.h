#pragma once
#include <chrono>
#include <unordered_map>
#include <string>
#include <iostream>
using std::chrono::milliseconds;
using namespace std;
/**
* A basic timer class.
*/
class Timer {
public:
	/**
	* Starts the timer
	*/
	inline void start() { t0 = std::chrono::steady_clock::now(); }

	/**
	* Stops the timer
	*/
	inline void stop() { t1 = std::chrono::steady_clock::now(); }

	/**
	* Return duration between the last call to start and last call to stop
	*/
	inline long	long duration() {
		return (std::chrono::duration_cast<milliseconds>(t1 - t0)).count();
	}
	void timeAvgStart(const string & type) {
		
		table[type].t0 = std::chrono::steady_clock::now();
	}
	int getCount(const string& type) { return table[type].count; }
	void printAvg(const string &type) {
		auto & avg = table[type];
		std::cout << type << "Count: " << (int)avg.count << " Average Time: " << (float)avg.totalDuration / (1000.0*avg.count) << "s." << endl;
	}
	void printCountRatio(const string& type1, const string& type2) {
		cout <<"avg count ratio"<< type2 <<" / " << type1 << (float)table[type2].count / (float)table[type1].count<<endl;
	}
	void timeAvgEnd(const string& type,int addcount=1) {
		//table[type].count++;
		table[type].count+=addcount;
		auto & avg = table[type];
		avg.t1 = std::chrono::steady_clock::now();
		avg.totalDuration += avg.duration();
		/*if (avg.count % 50==0)
			std::cout <<type<< "Count: "<<(int)avg.count<<" Average Time: "<<(float)avg.totalDuration/(1000.0*avg.count) <<"s."<<endl;*/

	}

	class AVGtimer {
	public:
		std::chrono::time_point<std::chrono::steady_clock> t0;
		std::chrono::time_point<std::chrono::steady_clock> t1;
		inline long	long duration() {
			return (std::chrono::duration_cast<milliseconds>(t1 - t0)).count();
		}
		int count = 0;
		long long totalDuration=0;
	};
private:
	std::chrono::time_point<std::chrono::steady_clock> t0;
	std::chrono::time_point<std::chrono::steady_clock> t1;
	std::unordered_map<string, AVGtimer> table;
};


