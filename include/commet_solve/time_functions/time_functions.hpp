#ifndef INCLUDE_TIME_FUNCTIONS_TIME_FUNCTIONS_HPP_
#define INCLUDE_TIME_FUNCTIONS_TIME_FUNCTIONS_HPP_

#include <utility>
#include <vector>

namespace commet_solve
{

using namespace std;

template <typename Number = double>
class TimeFunction
{
  public:
	TimeFunction() = default;
	virtual Number get_val(const Number &t) = 0;

  private:
};

template <typename Number = double>
class LinearRamp : public TimeFunction<Number>
{
  public:
	LinearRamp(const Number &end_value,
			   const Number &end_time,
			   const Number &start_value = 0,
			   const Number &start_time = 0)
		: TimeFunction<Number>()
		, end_value(end_value)
		, end_time(end_time)
		, start_value(start_value)
		, start_time(start_time) {};

	Number get_val(const Number &t) override
	{
		return start_value + (t - start_time) * (end_value - start_value) / (end_time - start_time);
	};

  private:
	const Number end_value;
	const Number end_time;
	const Number start_value;
	const Number start_time;
};

template <typename Number = double>
class DataFunction : public TimeFunction<Number>
{
  public:
	DataFunction(const vector<pair<Number, Number>> times_vals)
		: TimeFunction<Number>()
		, times_vals(times_vals)
		, start_time(times_vals.at(0).first)
		, end_time(times_vals.back().first) {};

	Number get_val(const Number &t) override
	{
		if (t <= this->times_vals.at(0).first)
			return this->times_vals.at(0).second;
		if (t >= this->times_vals.back().first)
			return this->times_vals.back().second;

		for (unsigned int i = 1; i < times_vals.size(); i++)
		{
			if (t <= times_vals.at(i).first)
			{
				const Number &t0 = times_vals.at(i - 1).first;
				const Number &t1 = times_vals.at(i).first;
				const Number &v0 = times_vals.at(i - 1).second;
				const Number &v1 = times_vals.at(i).second;
				return v0 + (t - t0) * (v1 - v0) / (t1 - t0);
			}
		}

		// return v0 + (t - t0) * (v1 - v0) / (t1 - t0);
		return this->times_vals.back().second;
	};

  private:
	const vector<pair<Number, Number>> times_vals;
	const Number start_time;
	const Number end_time;
};

} // namespace commet_solve

#endif // INCLUDE_TIME_FUNCTIONS_TIME_FUNCTIONS_HPP_
