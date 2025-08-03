#pragma once

#include "glass/graph.hpp"
#include "glass/neighbor.hpp"
#include "glass/quant/product_quant.hpp"
#include "glass/quant/quant.hpp"
#include "glass/quant/quant_base.hpp"
#include "glass/searcher/refiner.hpp"
#include "glass/searcher/searcher_base.hpp"
#include "glass/utils.hpp"
#include <omp.h>
#include <thread>

namespace glass {

template <QuantConcept Quant> struct GraphSearcherHighDim : public GraphSearcherBase {

  int32_t d;
  int32_t nb;
  Graph<int32_t> graph;
  Quant quant;

  // Search parameters
  int32_t ef = 32;

  // Memory prefetch parameters
  int32_t po = 1;
  int32_t pl = 1;
  int32_t graph_po = 1;

  // Enhanced processing parameters
  int32_t batch_size_factor = 2;
  int32_t early_stop_threshold = 8;

  // Optimization parameters
  constexpr static int32_t kOptimizePoints = 1000;
  constexpr static int32_t kTryPos = 20;
  constexpr static int32_t kTryPls = 20;
  constexpr static int32_t kTryK = 10;
  int32_t sample_points_num;
  std::vector<float> optimize_queries;

  mutable std::vector<
      LinearPool<typename Quant::ComputerType::dist_type, Bitset<>>>
      pools;

  GraphSearcherHighDim(Graph<int32_t> g)
      : graph(std::move(g)), graph_po(graph.K / 16),
        pools(std::thread::hardware_concurrency()) {}

  GraphSearcherHighDim(const GraphSearcherHighDim &) = delete;
  GraphSearcherHighDim(GraphSearcherHighDim &&) = delete;
  GraphSearcherHighDim &operator=(const GraphSearcherHighDim &) = delete;
  GraphSearcherHighDim &operator=(GraphSearcherHighDim &&) = delete;

  void SetData(const float *data, int32_t n, int32_t dim) override {
    this->nb = n;
    this->d = dim;
    quant = Quant(d);
    printf("Starting quantizer training\n");
    auto t1 = std::chrono::high_resolution_clock::now();
    quant.train(data, n);
    quant.add(data, n);
    auto t2 = std::chrono::high_resolution_clock::now();
    printf("Done quantizer training, cost %.2lfs\n",
           std::chrono::duration<double>(t2 - t1).count());

    sample_points_num = std::min(kOptimizePoints, nb - 1);
    std::vector<int32_t> sample_points(sample_points_num);
    std::mt19937 rng;
    GenRandom(rng, sample_points.data(), sample_points_num, nb);
    optimize_queries.resize((int64_t)sample_points_num * d);
    for (int32_t i = 0; i < sample_points_num; ++i) {
      memcpy(optimize_queries.data() + (int64_t)i * d,
             data + (int64_t)sample_points[i] * d, d * sizeof(float));
    }
  }

  void SetEf(int32_t ef) override { this->ef = ef; }

  int32_t GetEf() const override { return ef; }

  void Optimize(int32_t num_threads = 0) override {
    if (num_threads == 0) {
      num_threads = std::thread::hardware_concurrency();
    }
    std::vector<int32_t> try_pos(std::min(kTryPos, graph.K));
    std::vector<int32_t> try_pls(
        std::min(kTryPls, (int32_t)upper_div(quant.code_size(), 64)));
    std::iota(try_pos.begin(), try_pos.end(), 1);
    std::iota(try_pls.begin(), try_pls.end(), 1);
    std::vector<int32_t> dummy_dst(sample_points_num * kTryK);
    printf("=============Start optimization=============\n");
    auto timeit = [&] {
      auto st = std::chrono::high_resolution_clock::now();
      SearchBatch(optimize_queries.data(), sample_points_num, kTryK,
                  dummy_dst.data());
      auto ed = std::chrono::high_resolution_clock::now();
      return std::chrono::duration<double>(ed - st).count();
    };
    timeit(), timeit(), timeit(); // warmup
    float min_ela = std::numeric_limits<float>::max();
    int32_t best_po = 0, best_pl = 0;
    for (auto try_po : try_pos) {
      for (auto try_pl : try_pls) {
        this->po = try_po;
        this->pl = try_pl;
        auto ela = timeit();
        if (ela < min_ela) {
          min_ela = ela;
          best_po = try_po;
          best_pl = try_pl;
        }
      }
    }
    float baseline_ela;
    {
      this->po = 1;
      this->pl = 1;
      baseline_ela = timeit();
    }
    float slow_ela;
    {
      this->po = 0;
      this->pl = 0;
      slow_ela = timeit();
    }

    printf("settint best po = %d, best pl = %d\n"
           "gaining %6.2f%% performance improvement wrt baseline\ngaining "
           "%6.2f%% performance improvement wrt slow\n============="
           "Done optimization=============\n",
           best_po, best_pl, 100.0 * (baseline_ela / min_ela - 1),
           100.0 * (slow_ela / min_ela - 1));
    this->po = best_po;
    this->pl = best_pl;
    std::vector<float>().swap(optimize_queries);
  }

  void Search(const float *q, int32_t k, int32_t *dst) const override {
    auto computer = quant.get_computer(q);
    auto &pool = pools[omp_get_thread_num()];
    pool.reset(nb, std::max(k, ef), std::max(k, ef));
    graph.initialize_search(pool, computer);
    OptimizedSearchImpl(pool, computer);
    pool.to_sorted(dst, k);
  }

  mutable double last_search_avg_dist_cmps = 0.0;
  double GetLastSearchAvgDistCmps() const override {
    return last_search_avg_dist_cmps;
  }

  void SearchBatch(const float *qs, int32_t nq, int32_t k,
                   int32_t *dst) const override {
    std::atomic<int64_t> total_dist_cmps{0};
#pragma omp parallel for schedule(dynamic)
    for (int32_t i = 0; i < nq; ++i) {
      const float *cur_q = qs + i * d;
      int32_t *cur_dst = dst + i * k;
      auto computer = quant.get_computer(cur_q);
      auto &pool = pools[omp_get_thread_num()];
      pool.reset(nb, ef, std::max(k, ef));
      graph.initialize_search(pool, computer);
      OptimizedSearchImplBatch(pool, computer);
      pool.to_sorted(cur_dst, k);
      total_dist_cmps.fetch_add(computer.dist_cmps());
    }
    last_search_avg_dist_cmps = (double)total_dist_cmps.load() / nq;
  }

  // Optimized search implementation for single queries
  void OptimizedSearchImpl(NeighborPoolConcept auto &pool,
                          const ComputerConcept auto &computer) const {
    alignas(64) int32_t edge_buf[graph.K * 2]; // Larger buffer for better batching
    int32_t consecutive_no_improvement = 0;
    
    while (pool.has_next()) {
      auto u = pool.pop();
      
      // Enhanced prefetching with larger lookahead
      for (int32_t i = 0; i < std::min(po * batch_size_factor, graph.K); ++i) {
        int32_t to = graph.at(u, i);
        if (to == -1) break;
        if (!pool.check_visited(to)) {
          computer.prefetch(to, pl);
        }
      }
      
      int32_t improvements = 0;
      for (int32_t i = 0; i < graph.K; ++i) {
        int32_t v = graph.at(u, i);
        if (v == -1) break;
        
        // Continue prefetching ahead
        if (i + po * batch_size_factor < graph.K && graph.at(u, i + po * batch_size_factor) != -1) {
          int32_t to = graph.at(u, i + po * batch_size_factor);
          if (!pool.check_visited(to)) {
            computer.prefetch(to, pl);
          }
        }
        
        if (pool.check_visited(v)) continue;
        pool.set_visited(v);
        
        auto cur_dist = computer(v);
        if (pool.insert(v, cur_dist)) {
          graph.prefetch(v, graph_po);
          improvements++;
        }
      }
      
      // Smart early termination
      if (improvements == 0) {
        consecutive_no_improvement++;
        if (consecutive_no_improvement >= early_stop_threshold && pool.size() >= ef / 2) {
          break;
        }
      } else {
        consecutive_no_improvement = 0;
      }
    }
  }

  // Optimized search implementation for batch processing
  void OptimizedSearchImplBatch(NeighborPoolConcept auto &pool,
                               const ComputerConcept auto &computer) const {
    alignas(64) int32_t edge_buf[graph.K * 2];
    alignas(64) typename Quant::ComputerType::dist_type dist_buf[graph.K];
    
    while (pool.has_next()) {
      auto u = pool.pop();
      
      // Collect all valid edges first
      int32_t edge_size = 0;
      for (int32_t i = 0; i < graph.K; ++i) {
        int32_t v = graph.at(u, i);
        if (v == -1) break;
        if (pool.check_visited(v)) continue;
        pool.set_visited(v);
        edge_buf[edge_size++] = v;
      }
      
      if (edge_size == 0) continue;
      
      // Enhanced prefetching for the entire batch
      int32_t prefetch_batch = std::min(po * batch_size_factor, edge_size);
      for (int i = 0; i < prefetch_batch; ++i) {
        computer.prefetch(edge_buf[i], pl);
      }
      
      // Process edges in optimized batches
      for (int i = 0; i < edge_size; ++i) {
        // Continue prefetching ahead
        if (i + prefetch_batch < edge_size) {
          computer.prefetch(edge_buf[i + prefetch_batch], pl);
        }
        
        auto v = edge_buf[i];
        auto cur_dist = computer(v);
        
        if (pool.insert(v, cur_dist)) {
          graph.prefetch(v, graph_po);
        }
      }
    }
  }

  // Legacy implementations for compatibility
  void SearchImpl1(NeighborPoolConcept auto &pool,
                   const ComputerConcept auto &computer) const {
    OptimizedSearchImpl(pool, computer);
  }

  void SearchImpl2(NeighborPoolConcept auto &pool,
                   const ComputerConcept auto &computer) const {
    OptimizedSearchImplBatch(pool, computer);
  }
};

} // namespace glass
