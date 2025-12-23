#include <benchmark/benchmark.h>

#include "hilbert.h"
#include "hilbert_256_manual.h"
#include "hilbert_16.h"
#include "hilbert_cs.h"

void BM(benchmark::State& state){
	for(auto _ : state){
		auto table = make_table<16>();
		benchmark::DoNotOptimize(table);
	}
}


void BM_256_manual(benchmark::State& state){
	for(auto _ : state){
		auto table = make_table_256_manual<16>();
		benchmark::DoNotOptimize(table);
	}
}


void BM_16(benchmark::State& state){
	for(auto _ : state){
		auto table = make_table_16<16>();
		benchmark::DoNotOptimize(table);
	}
}

void BM_cs(benchmark::State& state){
	for(auto _ : state){
		auto table = make_table_cs<16>();
		benchmark::DoNotOptimize(table);
	}

}

size_t t {10};
size_t reps {4};

BENCHMARK(BM)->Unit(benchmark::kSecond)->MinTime(t)->Repetitions(reps)->ReportAggregatesOnly(true);
BENCHMARK(BM_256_manual)->Unit(benchmark::kSecond)->MinTime(t)->Repetitions(reps)->ReportAggregatesOnly(true);
BENCHMARK(BM_16)->Unit(benchmark::kSecond)->MinTime(t)->Repetitions(reps)->ReportAggregatesOnly(true);
BENCHMARK(BM_cs)->Unit(benchmark::kSecond)->MinTime(t)->Repetitions(reps)->ReportAggregatesOnly(true);

/*
BENCHMARK(BM)->Unit(benchmark::kSecond)->Iterations(3);
BENCHMARK(BM_256_manual)->Unit(benchmark::kSecond)->Iterations(3);
BENCHMARK(BM_16)->Unit(benchmark::kSecond)->Iterations(3);
*/

BENCHMARK_MAIN();
