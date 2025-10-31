#ifndef INCLUDE_COMMET_SOLVE_LOGGER_HPP_
#define INCLUDE_COMMET_SOLVE_LOGGER_HPP_

#include "config.hpp"

#include <chrono>
#include <ctime>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/mpi.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

namespace commet_solve
{
using namespace std;
using namespace dealii;

class Logger
{
  public:
	Logger()
		: mpi_communicator(MPI_COMM_WORLD)
		, pid(Utilities::MPI::this_mpi_process(mpi_communicator)) {};
	Logger(Logger &&) = default;
	Logger(const Logger &) = default;
	Logger &operator=(Logger &&) = default;
	Logger &operator=(const Logger &) = default;
	~Logger() {};
	void init()
	{
		this->mpi_communicator = MPI_Comm(MPI_COMM_WORLD);
		this->pid = Utilities::MPI::this_mpi_process(mpi_communicator);
	}

	void add_file(const string &path)
	{
		if (pid == 0)
			file_streams.push_back(make_unique<ofstream>(path));
	};

	void add_std_out()
	{
		console_streams.push_back(ConditionalOStream(cout, pid == 0));
	};

	inline void info(const string &msg)
	{
		const string prepped_message = "| INFO | " + time_in_HH_MM_SS_MMM() + " | " + msg + "\n";

		for (auto &stream : file_streams)
		{
			stream.get()->write(prepped_message.c_str(), prepped_message.size());
			stream.get()->flush();
		}

		for (auto &stream : console_streams)
		{
			stream << prepped_message;
		}
	}

	inline void debug(
#ifdef SHOW_DEBUG_MESSAGES
		const string &msg
#else
		const string & /*msg*/
#endif
	)
	{
#ifdef SHOW_DEBUG_MESSAGES
		const string prepped_message = "| DBUG | " + time_in_HH_MM_SS_MMM() + " | " + msg;

		for (auto &stream : file_streams)
		{
			stream.get()->write(prepped_message.c_str(), prepped_message.size());
			stream.get()->flush();
		}

		for (auto &stream : console_streams)
		{
			stream << prepped_message << endl;
		}
#endif
	}

  private:
	MPI_Comm mpi_communicator;
	unsigned int pid;
	// vector<ostream *> streams;
	vector<unique_ptr<ostream>> file_streams;
	vector<ConditionalOStream> console_streams;

	string time_in_HH_MM_SS_MMM()
	{
		using namespace std::chrono;

		// get current time
		auto now = system_clock::now();

		// get number of milliseconds for the current second
		// (remainder after division into seconds)
		auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;

		// convert to std::time_t in order to convert to std::tm (broken time)
		auto timer = system_clock::to_time_t(now);

		// convert to broken time
		std::tm bt = *std::localtime(&timer);

		std::ostringstream oss;

		oss << std::put_time(&bt, "%H:%M:%S"); // HH:MM:SS
		oss << '.' << std::setfill('0') << std::setw(3) << ms.count();

		return oss.str();
	}
};

static Logger LOGGER;


#define __FILENAME__ (__builtin_strrchr(__FILE__, '/') ? __builtin_strrchr(__FILE__, '/') + 1 : __FILE__)
#ifdef SHOW_DEBUG_MESSAGES
  #define DEBUG_MSG(msg) commet_solve::LOGGER.debug( std::string(__FILENAME__) + std::string(":") + std::to_string(__LINE__) + std::string(" ") + msg);
#else
  #define DEBUG_MSG(msg) do {} while(0);
#endif


} // namespace fs_mechanics

#endif // INCLUDE_COMMET_SOLVE_LOGGER_HPP_
