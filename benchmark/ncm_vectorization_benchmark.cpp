
#include "../include/commet_solve/parse_input/parse_ncm_materials.hpp"
#include "commet_solve/material_domain/ncm_domain/batch_vectorized_domain.hpp"
#include "commet_solve/material_domain/ncm_domain/globally_vectorized_domain.hpp"
#include "commet_solve/material_domain/ncm_domain/vectorized_domain.hpp"

#include "mointor_tools.hpp"

#include <benchmark/benchmark.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;
using namespace commet_benchmark;

struct Settings {
  unsigned int batch_size{1};
  unsigned int n_material_points{1};
  unsigned int ram_monitor_period_microseconds{1};
  commet_solve::NCMEvaluationMethod evaluation_method;
  std::string path_to_torchscript;
};

static Settings SETTINGS;

static void BM_batch_vectorized(benchmark::State &state) {
  PerfCounters counters;
  RAMUsageMonitor setup_ram_monitor;
  RAMUsageMonitor compute_ram_monitor;
  const int dim = 3;
  typedef double Number;

  setup_ram_monitor.start(SETTINGS.ram_monitor_period_microseconds);
  commet_solve::BatchVectorizedDomain<dim, Number> domain(SETTINGS.batch_size);
  domain.set_evaluation_method(SETTINGS.evaluation_method);
  domain.load_model(SETTINGS.path_to_torchscript);
  for (unsigned int i = 0; i < SETTINGS.n_material_points; i++)
    domain.add_entry(0, i);

  domain.close();
  setup_ram_monitor.stop();

  for (auto _ : state) {

    // PerfCounters counters;
    counters.start();
    compute_ram_monitor.start(SETTINGS.ram_monitor_period_microseconds);

    domain.compute_constitutive_behaviour();

    compute_ram_monitor.stop();
    counters.stop();
    counters.write_counters(state);

    state.counters["pre_setup_ram_usage_bytes"] = setup_ram_monitor.get_min_rss_bytes();
    state.counters["max_setup_ram_usage_bytes"] = setup_ram_monitor.get_max_rss_bytes();
    state.counters["pre_compute_ram_usage_bytes"] = compute_ram_monitor.get_min_rss_bytes();
    state.counters["max_compute_ram_usage_bytes"] = compute_ram_monitor.get_max_rss_bytes();
  }
}
BENCHMARK(BM_batch_vectorized);

static void BM_globally_vectorized(benchmark::State &state) {

  PerfCounters counters;
  RAMUsageMonitor setup_ram_monitor;
  RAMUsageMonitor compute_ram_monitor;
  const int dim = 3;
  typedef double Number;

  setup_ram_monitor.start(SETTINGS.ram_monitor_period_microseconds);
  commet_solve::GloballyVectorizedDomain<dim, Number> domain;
  domain.set_evaluation_method(SETTINGS.evaluation_method);
  domain.load_model(SETTINGS.path_to_torchscript);
  for (unsigned int i = 0; i < SETTINGS.n_material_points; i++)
    domain.add_entry(0, i);

  domain.close();
  setup_ram_monitor.stop();

  for (auto _ : state) {

    // PerfCounters counters;
    counters.start();
    compute_ram_monitor.start(SETTINGS.ram_monitor_period_microseconds);

    domain.compute_constitutive_behaviour();

    compute_ram_monitor.stop();
    counters.stop();
    counters.write_counters(state);

    state.counters["pre_setup_ram_usage_bytes"] = setup_ram_monitor.get_min_rss_bytes();
    state.counters["max_setup_ram_usage_bytes"] = setup_ram_monitor.get_max_rss_bytes();
    state.counters["pre_compute_ram_usage_bytes"] = compute_ram_monitor.get_min_rss_bytes();
    state.counters["max_compute_ram_usage_bytes"] = compute_ram_monitor.get_max_rss_bytes();
  }
}
BENCHMARK(BM_globally_vectorized);

// BENCHMARK_MAIN();
int main(int argc, char **argv) {

  torch::set_num_threads(1);
  std::string path_to_inp = std::string(argv[1]);
  json inp;
  std::ifstream(path_to_inp.c_str()) >> inp;

  if (inp.contains("path_to_torchscript"))
    SETTINGS.path_to_torchscript =
        inp["path_to_torchscript"].get<std::string>();

  if (inp.contains("batch_size"))
    SETTINGS.batch_size = inp["batch_size"].get<unsigned int>();

  if (inp.contains("n_material_points"))
    SETTINGS.n_material_points = inp["n_material_points"].get<unsigned int>();

  if (inp.contains("ram_monitor_period_microseconds"))
    SETTINGS.ram_monitor_period_microseconds =
        inp["ram_monitor_period_microseconds"].get<unsigned int>();

  SETTINGS.evaluation_method = commet_solve::parse::json_key_to_map_value(
      "evaluation_method", inp, commet_solve::parse::NCM_EVALUATION);

  char arg0_default[] = "benchmark";
  char *args_default = arg0_default;
  if (!argv) {
    argc = 1;
    argv = &args_default;
  }
  ::benchmark::Initialize(&argc, argv);
  ::benchmark::RunSpecifiedBenchmarks();
  ::benchmark::Shutdown();

  return 0;
}
int main(int, char **);
