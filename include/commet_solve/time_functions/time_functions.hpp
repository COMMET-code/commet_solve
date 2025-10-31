#ifndef INCLUDE_TIME_FUNCTIONS_TIME_FUNCTIONS_HPP_
#define INCLUDE_TIME_FUNCTIONS_TIME_FUNCTIONS_HPP_

#include <utility>
#include <vector>

namespace commet_solve
{

using namespace std;

template <typename ValueType = double, typename TimeType = double>
class TimeFunction
{
  public:
	TimeFunction() = default;
	virtual ValueType get_val(const TimeType &t) = 0;
	virtual ValueType get_increment(const TimeType &t, const TimeType &dt)
	{
		return this->get_val(t) - this->get_val(t - dt);
	};

	virtual TimeType get_start_time() const = 0;
	virtual TimeType get_end_time() const = 0;

  private:
};

template <typename ValueType = double, typename TimeType = double>
class LinearRamp : public TimeFunction<ValueType, TimeType>
{
  public:
	LinearRamp(const ValueType &end_value,
			   const TimeType &end_time,
			   const ValueType &start_value = 0,
			   const TimeType &start_time = 0)
		: TimeFunction<ValueType, TimeType>()
		, end_value(end_value)
		, start_value(start_value)
		, end_time(end_time)
		, start_time(start_time) {};

	ValueType get_val(const TimeType &t) override
	{
		return start_value + (t - start_time) * (end_value - start_value) / (end_time - start_time);
	};

	TimeType get_start_time() const override
	{
		return this->start_time;
	};
	TimeType get_end_time() const override
	{
		return this->end_time;
	};

  private:
	const ValueType end_value;
	const ValueType start_value;

	const TimeType end_time;
	const TimeType start_time;
};

template <typename ValueType = double, typename TimeType = double>
class DataFunction : public TimeFunction<ValueType, TimeType>
{
  public:
	DataFunction(const vector<pair<TimeType, ValueType>> &times_vals)
		: TimeFunction<ValueType, TimeType>()
		, times_vals(times_vals)
		, start_time(times_vals.front().first)
		, end_time(times_vals.back().first) {};

	ValueType get_val(const TimeType &t) override
	{
		const auto &tv = this->times_vals;
		if (t <= tv.front().first)
			return tv.front().second;
		if (t >= tv.back().first)
			return tv.back().second;

		// Binary search for the first element with time >= t
		auto it =
			std::lower_bound(tv.begin(), tv.end(), t, [](const std::pair<TimeType, ValueType> &p, const TimeType &val) {
				return p.first < val;
			});

		const auto &t1 = it->first;
		const auto &v1 = it->second;
		const auto &t0 = std::prev(it)->first;
		const auto &v0 = std::prev(it)->second;

		return v0 + (t - t0) * (v1 - v0) / (t1 - t0);
	}

	TimeType get_start_time() const override
	{
		return this->start_time;
	};
	TimeType get_end_time() const override
	{
		return this->end_time;
	};

  private:
	const vector<pair<TimeType, ValueType>> times_vals;
	const TimeType start_time;
	const TimeType end_time;
};

} // namespace commet_solve

#endif // INCLUDE_TIME_FUNCTIONS_TIME_FUNCTIONS_HPP_
