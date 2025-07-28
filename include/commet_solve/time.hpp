#ifndef INCLUDE_COMMET_SOLVE_TIME_HPP_
#define INCLUDE_COMMET_SOLVE_TIME_HPP_

#include "utilities.hpp"

namespace commet_solve {

template <typename Number = double>
class Time
{
  public:
	Time(const Number &end_time, const Number &delta_t)
		: timestep(0)
		, current_time(0.0)
		, end_time(end_time)
		, delta_t(delta_t) {};
	Time(Time<Number> &&other)
		: Time<Number>(other.end_time, other.delta_t) {
			// cout << "In move constructor" << endl;
		};
	Time(Time<Number> &other)
		: Time<Number>(other.end_time, other.delta_t) {
			// cout << "In move constructor" << endl;
		};

	Number current() const
	{
		return current_time;
	}

    void set_end(const Number & new_end){
        end_time = new_end;
    };

    void set_dt(const Number & new_dt){
        delta_t = new_dt;
    };

	Number end() const
	{
		return end_time;
	}

	unsigned int get_timestep() const
	{
		return timestep;
	}

	Number get_delta_t() const
	{
		return delta_t;
	}

	void increment()
	{
		current_time += delta_t;
		++timestep;
		if (current_time >= end_time || almost_equals(current_time, end_time))
		{
			current_time = end_time;
			at_end = true;
		}
		else
		{
			at_end = false;
		}
	};

	bool finished() const
	{
		return at_end;
	};

  private:
	unsigned int timestep;
	Number current_time;
	Number end_time;
	Number delta_t;
	bool at_end = false;

  protected:
};

} // namespace commet_solve

#endif // INCLUDE_COMMET_SOLVE_TIME_HPP_
