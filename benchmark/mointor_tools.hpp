#ifndef INCLUDE_BENCHMARK_MOINTOR_TOOLS_HPP_
#define INCLUDE_BENCHMARK_MOINTOR_TOOLS_HPP_

#include <asm/unistd.h>
#include <benchmark/benchmark.h>
#include <cstring>
#include <fstream>
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace commet_benchmark {

static int perf_event_open(struct perf_event_attr *hw_event, pid_t pid, int cpu,
                           int group_fd, unsigned long flags) {
  return syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
}

static long get_current_rss() {
  long rss = 0;
  std::ifstream statm("/proc/self/statm"); // statm is a file that contains
  long dummy = 0, resident_pages = 0;

  if (statm >> dummy >> resident_pages) { // skip first, read second
    rss = resident_pages * sysconf(_SC_PAGESIZE); //Final memory usage in bytes
  }

  return rss;
}

class RAMUsageMonitor {
  long start_rss{0};
  std::atomic<bool> running{false};
  std::atomic<long> max_rss_bytes{0};
  std::atomic<long> min_rss_bytes{0};
  std::thread monitor_thread;

public:
  void start(int interval_microseconds = 5) {
    running = true;
    start_rss = get_current_rss(); // initial value
    max_rss_bytes = start_rss;
    min_rss_bytes = start_rss;
    monitor_thread = std::thread([this, interval_microseconds]() {
      while (running) {
        long rss = get_current_rss();
        long prev_max = max_rss.load();
        while (rss > prev_max &&
               !max_rss.compare_exchange_weak(prev_max, rss)) {
          // loop until max_rss updated
        }

        long prev_min = min_rss.load();
        while (rss < prev_min &&
               !min_rss.compare_exchange_weak(prev_min, rss)) {
          // loop until min_rss updated
        }
        std::this_thread::sleep_for(
            std::chrono::microseconds(interval_microseconds));
      }
    });
  }

  void stop() {
    running = false;
    if (monitor_thread.joinable())
      monitor_thread.join();
  }
  ~RAMUsageMonitor() { this->stop(); }

  long get_max_rss_bytes() const { return max_rss_bytes.load(); }
  long get_min_rss_bytes() const { return min_rss_bytes.load(); }
  long get_start_rss_bytes() const { return start_rss; }
};

struct PerfCounters {
  int leader_fd = -1;
  int fd_cache_misses = -1;
  long long cache_misses_{0};
  long long instructions_{0};

  PerfCounters() {
    struct perf_event_attr pe{};
    memset(&pe, 0, sizeof(pe));
    pe.size = sizeof(pe);
    pe.disabled = 1;
    pe.exclude_kernel = 1;
    pe.exclude_hv = 1;

    // Leader: total instructions
    pe.type = PERF_TYPE_HARDWARE;
    pe.config = PERF_COUNT_HW_INSTRUCTIONS;
    leader_fd = perf_event_open(&pe, 0, -1, -1, 0);
    if (leader_fd == -1)
      perror("leader perf_event_open");

    // Group: cache misses
    pe.disabled = 0;
    pe.config = PERF_COUNT_HW_CACHE_MISSES;
    fd_cache_misses = perf_event_open(&pe, 0, -1, leader_fd, 0);
  }

  void start() {
    ioctl(leader_fd, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP);
    ioctl(leader_fd, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP);
  }

  void stop() { ioctl(leader_fd, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP); }

  // void read_counts(long long &instructions, long long &cache_misses) {
  void read_counts() {
    long long n = read(leader_fd, &instructions_, sizeof(instructions_));
    if (n != sizeof(instructions_)) {
      perror("error reading number of instructions");
    }

    n = read(fd_cache_misses, &cache_misses_, sizeof(cache_misses_));
    if (n != sizeof(cache_misses_)) {
      perror("error reading number of cache misses");
    }
  }

  void write_counters(benchmark::State &state) {
    this->read_counts();
    state.counters["instructions"] = instructions_;
    state.counters["cache_misses"] = cache_misses_;
  };

  ~PerfCounters() {
    if (leader_fd != -1)
      close(leader_fd);
    if (fd_cache_misses != -1)
      close(fd_cache_misses);
  }
};

} // namespace commet_benchmark

#endif // INCLUDE_BENCHMARK_MOINTOR_TOOLS_HPP_
