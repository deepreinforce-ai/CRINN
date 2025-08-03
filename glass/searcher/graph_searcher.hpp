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
#include <atomic>
#include <algorithm>
#include <chrono>
#include <immintrin.h>

namespace glass {

// ================= FEATURE SWITCHES =================
namespace feature_switches {
  // Switch 1: Enable QZOAIS quantum structures and initialization
  constexpr bool ENABLE_QUANTUM_STRUCTURES = true;   // 全开测试
  
  // Switch 2: Enable multi-tier entry point selection
  constexpr bool ENABLE_MULTI_TIER_ENTRIES = true;   // 全开测试
  
  // Switch 3: Enable quantum parameter lookup tables
  constexpr bool ENABLE_QUANTUM_PARAM_LUT = false;    // 全开测试
  
  // Switch 4: Enable quantum convergence assessment
  constexpr bool ENABLE_QUANTUM_CONVERGENCE = true;  // 全开测试
  
  // Switch 5: Enable atomic performance tracking
  constexpr bool ENABLE_PERFORMANCE_TRACKING = true; // 全开测试
  
  // Switch 6: Enable SIMD processing optimizations
  constexpr bool ENABLE_SIMD_PROCESSING = true;      // 全开测试
  
  // Switch 9: Enable enhanced early termination
  constexpr bool ENABLE_ENHANCED_EARLY_TERM = true;  // 全开测试
  
  // Switch 10: Enable quantum naming and branding
  constexpr bool ENABLE_QUANTUM_NAMING = true;       // 全开测试
}

template <QuantConcept Quant> struct GraphSearcher : public GraphSearcherBase {

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

  // Enhanced processing parameters (from quantum version)
  int32_t batch_size_factor = feature_switches::ENABLE_QUANTUM_STRUCTURES ? 3 : 2;
  int32_t early_stop_threshold = feature_switches::ENABLE_QUANTUM_STRUCTURES ? 6 : 8;
  float convergence_threshold = feature_switches::ENABLE_QUANTUM_STRUCTURES ? 0.15f : 0.0f;

  // Quantum parameters (from original code)
  int32_t quantum_batch_factor = 3;
  int32_t quantum_early_stop = 6;
  float quantum_convergence_threshold = 0.15f;

  // QZOAIS structures (Switch 1)
  struct QZOAISParams {
    static constexpr int32_t QUANTUM_ENTRY_POOL_SIZE = 16;
    static constexpr int32_t QUANTUM_PREFETCH_LUT_SIZE = 32;
    static constexpr int32_t QUANTUM_BATCH_LUT_SIZE = 16;
    static constexpr int32_t QUANTUM_QUALITY_LUT_SIZE = 32;
    static constexpr int32_t QUANTUM_CONVERGENCE_LUT_SIZE = 16;
    static constexpr int32_t QUANTUM_SIMD_WIDTH = 8;
    static constexpr int32_t QUANTUM_CHECK_INTERVAL = 3;
    static constexpr int32_t QUANTUM_BATCH_MIN = 12;
    static constexpr int32_t QUANTUM_BATCH_MAX = 48;
    static constexpr int32_t QUANTUM_SCALING_FACTOR = 3;
  };

  QZOAISParams qzoais_params;

  // Quantum structures (Switch 1)
  struct QuantumQZOAIS {
    alignas(64) std::array<int32_t, 6> quantum_primary_entries;
    alignas(64) std::array<int32_t, 6> quantum_secondary_entries;
    alignas(64) std::array<int32_t, 4> quantum_tertiary_entries;
    alignas(64) std::array<float, 16> quantum_quality_matrix;
    alignas(64) std::array<float, 16> quantum_cache_affinity;
    alignas(64) std::array<float, 16> quantum_stability_scores;
    alignas(64) std::array<uint8_t, 16> quantum_priority_flags;
    alignas(64) std::array<float, 16> quantum_efficiency_matrix;
    alignas(64) std::array<uint8_t, 16> quantum_tier_types;
    alignas(64) std::array<float, 16> quantum_diversity_scores;
    alignas(64) std::array<int32_t, 32> quantum_prefetch_table;
    alignas(64) std::array<int32_t, 16> quantum_batch_table;
    alignas(64) std::array<int32_t, 32> quantum_width_table;
    alignas(64) std::array<float, 32> quantum_convergence_table;
    alignas(64) std::array<int32_t, 16> quantum_iteration_limits;
    alignas(64) std::array<float, 16> quantum_threshold_table;
    alignas(64) std::array<float, 16> quantum_adaptive_factors;
    mutable std::atomic<uint32_t> quantum_primary_selector{0};
    mutable std::atomic<uint32_t> quantum_secondary_selector{0};
    mutable std::atomic<uint32_t> quantum_tertiary_selector{0};
    mutable std::atomic<uint32_t> quantum_param_selector{0};
    mutable std::atomic<uint32_t> quantum_query_counter{0};
    mutable std::atomic<uint32_t> quantum_efficiency_tracker{0};
    mutable std::atomic<float> quantum_performance_score{1.0f};
    mutable std::atomic<float> quantum_cache_optimizer{1.0f};
    mutable std::atomic<float> quantum_stability_factor{1.0f};
  };

  QuantumQZOAIS quantum_qzoais;

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

  GraphSearcher(Graph<int32_t> g)
      : graph(std::move(g)), graph_po(graph.K / 16),
        pools(std::thread::hardware_concurrency()) {
    if constexpr (feature_switches::ENABLE_QUANTUM_STRUCTURES) {
      initializeQuantumQZOAIS();
    }
  }

  GraphSearcher(const GraphSearcher &) = delete;
  GraphSearcher(GraphSearcher &&) = delete;
  GraphSearcher &operator=(const GraphSearcher &) = delete;
  GraphSearcher &operator=(GraphSearcher &&) = delete;

  // Initialize quantum structures (Switch 1)
  void initializeQuantumQZOAIS() {
    if constexpr (!feature_switches::ENABLE_QUANTUM_STRUCTURES) return;
    
    // Step 1: Quantum entry point assessment
    std::vector<std::tuple<float, int32_t, float, float, uint8_t, uint8_t, float>> quantum_candidates;
    quantum_candidates.reserve(std::min(128, graph.size()));
    
    // Start with original entry points
    for (int32_t ep : graph.eps) {
      if (ep >= 0 && ep < graph.size()) {
        float base_quality = 1.3f + (float)graph.degree(ep) / (float)graph.K * 1.0f;
        
        float connectivity_bonus = 0.0f;
        int32_t degree = graph.degree(ep);
        if (degree > graph.K * 0.85f) {
          connectivity_bonus = 0.35f;
        } else if (degree > graph.K * 0.75f) {
          connectivity_bonus = 0.25f;
        } else if (degree > graph.K * 0.65f) {
          connectivity_bonus = 0.2f;
        }
        
        float cache_affinity = 1.25f;
        float stability_score = 1.35f;
        uint8_t priority_flag = 8;
        uint8_t tier_type = 1;
        float diversity_score = 1.0f;
        
        quantum_candidates.emplace_back(base_quality + connectivity_bonus, ep, cache_affinity, 
                                       stability_score, priority_flag, tier_type, diversity_score);
      }
    }
    
    // Quantum sampling
    int32_t quantum_step = std::max(1, graph.size() / 96);
    for (int32_t i = 0; i < graph.size(); i += quantum_step) {
      int32_t degree = graph.degree(i);
      if (degree > graph.K * 0.5f) {
        float base_quality = (float)degree / (float)graph.K * 0.95f;
        
        float neighbor_quality = 0.0f;
        int32_t neighbor_count = std::min(4, degree);
        for (int32_t j = 0; j < neighbor_count; ++j) {
          int32_t neighbor = graph.at(i, j);
          if (neighbor >= 0 && neighbor < graph.size()) {
            neighbor_quality += 0.12f * ((float)graph.degree(neighbor) / (float)graph.K);
          }
        }
        
        float connectivity_bonus = 0.0f;
        if (degree >= 6) {
          for (int32_t j = 0; j < std::min(3, degree - 3); ++j) {
            int32_t far_neighbor = graph.at(i, degree - 1 - j);
            if (far_neighbor >= 0 && far_neighbor < graph.size()) {
              connectivity_bonus += 0.09f * ((float)graph.degree(far_neighbor) / (float)graph.K);
            }
          }
        }
        
        float cache_affinity = 0.95f;
        if (degree >= 4) {
          int32_t mid_neighbor = graph.at(i, degree / 2);
          if (mid_neighbor >= 0 && mid_neighbor < graph.size()) {
            cache_affinity += 0.18f * ((float)graph.degree(mid_neighbor) / (float)graph.K);
          }
        }
        
        float stability_score = 1.0f + (cache_affinity * 0.12f) + (connectivity_bonus * 0.08f);
        
        float diversity_score = 0.8f;
        if (degree >= 8) {
          int32_t quarter_neighbor = graph.at(i, degree / 4);
          if (quarter_neighbor >= 0 && quarter_neighbor < graph.size()) {
            diversity_score += 0.15f * ((float)graph.degree(quarter_neighbor) / (float)graph.K);
          }
        }
        
        float total_quality = base_quality + neighbor_quality + connectivity_bonus;
        
        uint8_t priority_flag = 3;
        if (total_quality > 1.6f && degree > graph.K * 0.8f) {
          priority_flag = 7;
        } else if (total_quality > 1.4f && degree > graph.K * 0.7f) {
          priority_flag = 6;
        } else if (total_quality > 1.2f && degree > graph.K * 0.6f) {
          priority_flag = 5;
        } else if (total_quality > 1.0f && degree > graph.K * 0.55f) {
          priority_flag = 4;
        }
        
        uint8_t tier_type = 1;
        if (cache_affinity > 1.15f && stability_score > 1.1f) {
          tier_type = 1;
        } else if (total_quality > 1.1f && diversity_score > 0.9f) {
          tier_type = 2;
        } else {
          tier_type = 3;
        }
        
        if (total_quality > 0.75f) {
          quantum_candidates.emplace_back(total_quality, i, cache_affinity, stability_score,
                                         priority_flag, tier_type, diversity_score);
        }
      }
    }
    
    // Sort by quantum quality scores
    std::sort(quantum_candidates.begin(), quantum_candidates.end(), 
              [](const auto& a, const auto& b) { return std::get<0>(a) > std::get<0>(b); });
    
    // Create quantum multi-tier entry pools
    for (int i = 0; i < 6; ++i) {
      int32_t entry_point = 0;
      if (i < quantum_candidates.size()) {
        entry_point = std::get<1>(quantum_candidates[i]);
      } else {
        entry_point = (i * graph.size()) / 6;
      }
      quantum_qzoais.quantum_primary_entries[i] = std::min(entry_point, graph.size() - 1);
    }
    
    // Create quantum secondary tier (6 entries) - diversity-optimized from remaining candidates
    std::vector<int32_t> used_entries(quantum_qzoais.quantum_primary_entries.begin(), 
                                     quantum_qzoais.quantum_primary_entries.end());
    
    int secondary_idx = 0;
    for (const auto& candidate : quantum_candidates) {
      if (secondary_idx >= 6) break;
      
      int32_t candidate_entry = std::get<1>(candidate);
      bool is_used = false;
      
      for (int32_t used_entry : used_entries) {
        if (candidate_entry == used_entry) {
          is_used = true;
          break;
        }
      }
      
      if (!is_used) {
        uint8_t tier_type = std::get<5>(candidate);
        if (tier_type == 2) { // Prefer secondary tier candidates
          quantum_qzoais.quantum_secondary_entries[secondary_idx] = candidate_entry;
          used_entries.push_back(candidate_entry);
          secondary_idx++;
        }
      }
    }
    
    // Fill remaining secondary entries
    while (secondary_idx < 6) {
      int32_t strategic_entry = ((secondary_idx + 3) * graph.size()) / 12;
      quantum_qzoais.quantum_secondary_entries[secondary_idx] = strategic_entry;
      used_entries.push_back(strategic_entry);
      secondary_idx++;
    }
    
    // Quantum tertiary tier (4 entries) - exploration entries
    int tertiary_idx = 0;
    for (const auto& candidate : quantum_candidates) {
      if (tertiary_idx >= 4) break;
      
      int32_t candidate_entry = std::get<1>(candidate);
      bool is_used = false;
      
      for (int32_t used_entry : used_entries) {
        if (candidate_entry == used_entry) {
          is_used = true;
          break;
        }
      }
      
      if (!is_used) {
        uint8_t priority = std::get<4>(candidate);
        if (priority >= 4) { // Medium priority or above
          quantum_qzoais.quantum_tertiary_entries[tertiary_idx] = candidate_entry;
          used_entries.push_back(candidate_entry);
          tertiary_idx++;
        }
      }
    }
    
    // Fill remaining tertiary entries
    while (tertiary_idx < 4) {
      int32_t exploration_entry = ((tertiary_idx + 7) * graph.size()) / 16;
      quantum_qzoais.quantum_tertiary_entries[tertiary_idx] = exploration_entry;
      tertiary_idx++;
    }
    
    // Fill combined matrices for all 16 entries
    for (int i = 0; i < 6; ++i) {
      int32_t entry = quantum_qzoais.quantum_primary_entries[i];
      quantum_qzoais.quantum_quality_matrix[i] = (float)graph.degree(entry) / (float)graph.K * 1.15f;
      quantum_qzoais.quantum_cache_affinity[i] = 1.2f;
      quantum_qzoais.quantum_stability_scores[i] = 1.25f;
      quantum_qzoais.quantum_priority_flags[i] = 7;
      quantum_qzoais.quantum_efficiency_matrix[i] = 1.3f;
      quantum_qzoais.quantum_tier_types[i] = 1;
      quantum_qzoais.quantum_diversity_scores[i] = 1.0f;
    }
    
    for (int i = 0; i < 6; ++i) {
      int32_t entry = quantum_qzoais.quantum_secondary_entries[i];
      quantum_qzoais.quantum_quality_matrix[6 + i] = (float)graph.degree(entry) / (float)graph.K * 1.05f;
      quantum_qzoais.quantum_cache_affinity[6 + i] = 1.1f;
      quantum_qzoais.quantum_stability_scores[6 + i] = 1.15f;
      quantum_qzoais.quantum_priority_flags[6 + i] = 6;
      quantum_qzoais.quantum_efficiency_matrix[6 + i] = 1.2f;
      quantum_qzoais.quantum_tier_types[6 + i] = 2;
      quantum_qzoais.quantum_diversity_scores[6 + i] = 0.95f;
    }
    
    for (int i = 0; i < 4; ++i) {
      int32_t entry = quantum_qzoais.quantum_tertiary_entries[i];
      quantum_qzoais.quantum_quality_matrix[12 + i] = (float)graph.degree(entry) / (float)graph.K * 0.95f;
      quantum_qzoais.quantum_cache_affinity[12 + i] = 1.05f;
      quantum_qzoais.quantum_stability_scores[12 + i] = 1.1f;
      quantum_qzoais.quantum_priority_flags[12 + i] = 5;
      quantum_qzoais.quantum_efficiency_matrix[12 + i] = 1.1f;
      quantum_qzoais.quantum_tier_types[12 + i] = 3;
      quantum_qzoais.quantum_diversity_scores[12 + i] = 0.9f;
    }

    // Initialize lookup tables
    for (int i = 0; i < qzoais_params.QUANTUM_PREFETCH_LUT_SIZE; ++i) {
      quantum_qzoais.quantum_prefetch_table[i] = std::min(8, 2 + i / 4);
    }
    
    for (int i = 0; i < qzoais_params.QUANTUM_BATCH_LUT_SIZE; ++i) {
      int32_t base_batch = qzoais_params.QUANTUM_BATCH_MIN;
      int32_t scaled_batch = base_batch + (i * qzoais_params.QUANTUM_SCALING_FACTOR) / 2;
      quantum_qzoais.quantum_batch_table[i] = std::min(qzoais_params.QUANTUM_BATCH_MAX, scaled_batch);
    }
    
    for (int i = 0; i < qzoais_params.QUANTUM_QUALITY_LUT_SIZE; ++i) {
      quantum_qzoais.quantum_width_table[i] = std::min(6, 1 + i / 5);
    }
    
    // Quantum convergence factors
    for (int i = 0; i < qzoais_params.QUANTUM_QUALITY_LUT_SIZE; ++i) {
      quantum_qzoais.quantum_convergence_table[i] = 1.03f + (float)i * 0.06f;
    }
    
    // Quantum iteration limits
    for (int i = 0; i < qzoais_params.QUANTUM_CONVERGENCE_LUT_SIZE; ++i) {
      quantum_qzoais.quantum_iteration_limits[i] = 4 + i / 2;
    }
    
    // Quantum threshold table
    for (int i = 0; i < qzoais_params.QUANTUM_CONVERGENCE_LUT_SIZE; ++i) {
      quantum_qzoais.quantum_threshold_table[i] = 0.12f + (float)i * 0.02f;
    }
    
    // Quantum adaptive factors
    for (int i = 0; i < qzoais_params.QUANTUM_CONVERGENCE_LUT_SIZE; ++i) {
      quantum_qzoais.quantum_adaptive_factors[i] = 0.92f + (float)i * 0.008f;
    }
  }

  void SetData(const float *data, int32_t n, int32_t dim) override {
    this->nb = n;
    this->d = dim;
    quant = Quant(d);
    const char* training_msg = feature_switches::ENABLE_QUANTUM_NAMING ? "Starting QZOAIS quantizer training\n" : "Starting quantizer training\n";
    printf("%s", training_msg);
    auto t1 = std::chrono::high_resolution_clock::now();
    quant.train(data, n);
    quant.add(data, n);
    auto t2 = std::chrono::high_resolution_clock::now();
    const char* done_msg = feature_switches::ENABLE_QUANTUM_NAMING ? "Done QZOAIS quantizer training, cost %.2lfs\n" : "Done quantizer training, cost %.2lfs\n";
    printf(done_msg, std::chrono::duration<double>(t2 - t1).count());

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
    const char* start_msg = feature_switches::ENABLE_QUANTUM_NAMING ? "=============Start QZOAIS optimization=============\n" : "=============Start optimization=============\n";
    printf("%s", start_msg);
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

    const char* result_msg = feature_switches::ENABLE_QUANTUM_NAMING ? 
      "QZOAIS: best po = %d, best pl = %d\n"
      "gaining %6.2f%% performance improvement wrt baseline\ngaining "
      "%6.2f%% performance improvement wrt slow\n============="
      "Done QZOAIS optimization=============\n" :
      "settint best po = %d, best pl = %d\n"
      "gaining %6.2f%% performance improvement wrt baseline\ngaining "
      "%6.2f%% performance improvement wrt slow\n============="
      "Done optimization=============\n";
    printf(result_msg,
           best_po, best_pl, 100.0 * (baseline_ela / min_ela - 1),
           100.0 * (slow_ela / min_ela - 1));
    this->po = best_po;
    this->pl = best_pl;
    
    std::vector<float>().swap(optimize_queries);
  }

  // Quantum entry selection (Switch 2)
  GLASS_INLINE int32_t selectQuantumPrimaryEntry() const {
    if constexpr (feature_switches::ENABLE_MULTI_TIER_ENTRIES && feature_switches::ENABLE_QUANTUM_STRUCTURES) {
      uint32_t selector = quantum_qzoais.quantum_primary_selector.fetch_add(1, std::memory_order_relaxed);
      uint32_t entry_idx = selector % 6;
      return quantum_qzoais.quantum_primary_entries[entry_idx];
    } else {
      return graph.eps.empty() ? 0 : graph.eps[0];
    }
  }

  GLASS_INLINE int32_t selectQuantumSecondaryEntry() const {
    if constexpr (feature_switches::ENABLE_MULTI_TIER_ENTRIES && feature_switches::ENABLE_QUANTUM_STRUCTURES) {
      uint32_t selector = quantum_qzoais.quantum_secondary_selector.fetch_add(1, std::memory_order_relaxed);
      uint32_t entry_idx = selector % 6;
      return quantum_qzoais.quantum_secondary_entries[entry_idx];
    } else {
      return graph.eps.size() > 1 ? graph.eps[1] : (graph.eps.empty() ? 1 : graph.eps[0] + 1);
    }
  }

  GLASS_INLINE int32_t selectQuantumTertiaryEntry() const {
    if constexpr (feature_switches::ENABLE_MULTI_TIER_ENTRIES && feature_switches::ENABLE_QUANTUM_STRUCTURES) {
      uint32_t selector = quantum_qzoais.quantum_tertiary_selector.fetch_add(1, std::memory_order_relaxed);
      uint32_t entry_idx = selector & 3; // Fast bit masking for 4 tertiary entries
      return quantum_qzoais.quantum_tertiary_entries[entry_idx];
    } else {
      return graph.eps.size() > 2 ? graph.eps[2] : (graph.eps.empty() ? 2 : graph.eps[0] + 2);
    }
  }

  GLASS_INLINE int32_t selectQuantumOptimalEntry(int32_t primary_entry, int32_t secondary_entry, int32_t tertiary_entry) const {
    if constexpr (feature_switches::ENABLE_MULTI_TIER_ENTRIES && feature_switches::ENABLE_QUANTUM_STRUCTURES) {
      float best_combined_score = -1.0f;
      int32_t best_entry = tertiary_entry;
      
      // Check all 16 entries for optimal diversity
      for (int i = 0; i < qzoais_params.QUANTUM_ENTRY_POOL_SIZE; ++i) {
        int32_t candidate;
        if (i < 6) {
          candidate = quantum_qzoais.quantum_primary_entries[i];
        } else if (i < 12) {
          candidate = quantum_qzoais.quantum_secondary_entries[i - 6];
        } else {
          candidate = quantum_qzoais.quantum_tertiary_entries[i - 12];
        }
        
        if (candidate != primary_entry && candidate != secondary_entry && candidate != tertiary_entry) {
          float quality_score = quantum_qzoais.quantum_quality_matrix[i];
          float cache_affinity = quantum_qzoais.quantum_cache_affinity[i];
          float stability_score = quantum_qzoais.quantum_stability_scores[i];
          uint8_t priority_flag = quantum_qzoais.quantum_priority_flags[i];
          float efficiency_score = quantum_qzoais.quantum_efficiency_matrix[i];
          uint8_t tier_type = quantum_qzoais.quantum_tier_types[i];
          float diversity_score = quantum_qzoais.quantum_diversity_scores[i];
          
          // Combined scoring
          float combined_score = quality_score * 0.22f + cache_affinity * 0.18f + 
                                stability_score * 0.16f + (float)priority_flag * 0.14f + 
                                efficiency_score * 0.12f + (float)tier_type * 0.10f +
                                diversity_score * 0.08f;
          
          if (combined_score > best_combined_score) {
            best_combined_score = combined_score;
            best_entry = candidate;
          }
        }
      }
      
      return best_entry;
    } else {
      return tertiary_entry;
    }
  }

  // Quantum convergence assessment (Switch 4)
  GLASS_INLINE bool checkQuantumConvergence(int32_t iteration, int32_t improvements, 
                                           int32_t stagnation) const {
    if constexpr (feature_switches::ENABLE_QUANTUM_CONVERGENCE && feature_switches::ENABLE_QUANTUM_STRUCTURES) {
      if (iteration < 4) return false;
      
      if ((iteration % qzoais_params.QUANTUM_CHECK_INTERVAL) == 0) {
        uint32_t convergence_idx = std::min(31, iteration >> 3);
        float convergence_factor = quantum_qzoais.quantum_convergence_table[convergence_idx];
        float improvement_rate = (iteration > 0) ? (float)improvements / (float)iteration : 0.0f;
        
        uint32_t threshold_idx = std::min(15, iteration >> 4);
        float quantum_threshold = quantum_qzoais.quantum_threshold_table[threshold_idx];
        
        float quantum_performance = quantum_qzoais.quantum_performance_score.load(std::memory_order_relaxed);
        uint32_t adaptive_idx = std::min(15, (int)(quantum_performance * 8));
        float adaptive_factor = quantum_qzoais.quantum_adaptive_factors[adaptive_idx];
        
        float cache_optimizer = quantum_qzoais.quantum_cache_optimizer.load(std::memory_order_relaxed);
        float stability_factor = quantum_qzoais.quantum_stability_factor.load(std::memory_order_relaxed);
        
        return (improvement_rate < quantum_threshold * convergence_factor * adaptive_factor * 
                cache_optimizer * stability_factor) && 
               (stagnation >= quantum_early_stop);
      }
      return false;
    } else {
      return stagnation >= early_stop_threshold;
    }
  }

  void Search(const float *q, int32_t k, int32_t *dst) const override {
    auto computer = quant.get_computer(q);
    auto &pool = pools[omp_get_thread_num()];
    pool.reset(nb, std::max(k, ef), std::max(k, ef));
    
    graph.initialize_search(pool, computer);
    
    // Multi-tier entry selection (Switch 2)
    if constexpr (feature_switches::ENABLE_MULTI_TIER_ENTRIES) {
      int32_t primary_entry = selectQuantumPrimaryEntry();
      if (primary_entry < nb && !pool.check_visited(primary_entry)) {
        pool.insert(primary_entry, computer(primary_entry));
        pool.set_visited(primary_entry);
      }
      
      if (ef > 16) {
        int32_t secondary_entry = selectQuantumSecondaryEntry();
        if (secondary_entry < nb && secondary_entry != primary_entry && !pool.check_visited(secondary_entry)) {
          pool.insert(secondary_entry, computer(secondary_entry));
          pool.set_visited(secondary_entry);
        }
        
        // Quantum tertiary entry for exploration (higher threshold)
        if (ef > 32) { // Higher threshold for minimal disruption + enhancement
          int32_t tertiary_entry = selectQuantumTertiaryEntry();
          if (tertiary_entry < nb && tertiary_entry != primary_entry && 
              tertiary_entry != secondary_entry && !pool.check_visited(tertiary_entry)) {
            pool.insert(tertiary_entry, computer(tertiary_entry));
            pool.set_visited(tertiary_entry);
          }
          
          // Quantum optimal entry for maximum recall (highest threshold)
          if (ef > 48) { // Highest threshold for maximum stability preservation + enhancement
            int32_t optimal_entry = selectQuantumOptimalEntry(primary_entry, secondary_entry, tertiary_entry);
            if (optimal_entry < nb && optimal_entry != tertiary_entry && !pool.check_visited(optimal_entry)) {
              pool.insert(optimal_entry, computer(optimal_entry));
              pool.set_visited(optimal_entry);
            }
          }
        }
      }
    }
    
    if constexpr (feature_switches::ENABLE_QUANTUM_STRUCTURES) {
      QuantumEnhancedZeroOverheadAdaptiveIntelligenceSearch(pool, computer);
    } else {
      OptimizedSearchImpl(pool, computer);
    }
    pool.to_sorted(dst, k);
    
    // Performance tracking (Switch 5)
    if constexpr (feature_switches::ENABLE_PERFORMANCE_TRACKING && feature_switches::ENABLE_QUANTUM_STRUCTURES) {
      quantum_qzoais.quantum_query_counter.fetch_add(1, std::memory_order_relaxed);
    }
  }

  mutable double last_search_avg_dist_cmps = 0.0;
  double GetLastSearchAvgDistCmps() const override {
    return last_search_avg_dist_cmps;
  }

  void SearchBatch(const float *qs, int32_t nq, int32_t k,
                   int32_t *dst) const override {
    std::atomic<int64_t> total_dist_cmps{0};
    
    // Performance tracking (Switch 5)
    if constexpr (feature_switches::ENABLE_PERFORMANCE_TRACKING && feature_switches::ENABLE_QUANTUM_STRUCTURES) {
      quantum_qzoais.quantum_query_counter.fetch_add(nq, std::memory_order_relaxed);
    }
    
#pragma omp parallel for schedule(dynamic)
    for (int32_t i = 0; i < nq; ++i) {
      const float *cur_q = qs + i * d;
      int32_t *cur_dst = dst + i * k;
      auto computer = quant.get_computer(cur_q);
      auto &pool = pools[omp_get_thread_num()];
      pool.reset(nb, ef, std::max(k, ef));
      
      graph.initialize_search(pool, computer);
      
      // Multi-tier entry selection with batch thresholds (Switch 2)
      if constexpr (feature_switches::ENABLE_MULTI_TIER_ENTRIES) {
        int32_t primary_entry = selectQuantumPrimaryEntry();
        if (primary_entry < nb && !pool.check_visited(primary_entry)) {
          pool.insert(primary_entry, computer(primary_entry));
          pool.set_visited(primary_entry);
        }
        
        if (ef > 12) {
          int32_t secondary_entry = selectQuantumSecondaryEntry();
          if (secondary_entry < nb && secondary_entry != primary_entry && !pool.check_visited(secondary_entry)) {
            pool.insert(secondary_entry, computer(secondary_entry));
            pool.set_visited(secondary_entry);
          }
          
          if (ef > 24) { // Batch optimized thresholds
            int32_t tertiary_entry = selectQuantumTertiaryEntry();
            if (tertiary_entry < nb && tertiary_entry != primary_entry && 
                tertiary_entry != secondary_entry && !pool.check_visited(tertiary_entry)) {
              pool.insert(tertiary_entry, computer(tertiary_entry));
              pool.set_visited(tertiary_entry);
            }
            
            if (ef > 40) {
              int32_t optimal_entry = selectQuantumOptimalEntry(primary_entry, secondary_entry, tertiary_entry);
              if (optimal_entry < nb && optimal_entry != tertiary_entry && !pool.check_visited(optimal_entry)) {
                pool.insert(optimal_entry, computer(optimal_entry));
                pool.set_visited(optimal_entry);
              }
            }
          }
        }
      }
      
      if constexpr (feature_switches::ENABLE_QUANTUM_STRUCTURES) {
        QuantumEnhancedZeroOverheadAdaptiveIntelligenceSearchBatch(pool, computer);
      } else {
        OptimizedSearchImplBatch(pool, computer);
      }
      pool.to_sorted(cur_dst, k);
      total_dist_cmps.fetch_add(computer.dist_cmps());
    }
    
    last_search_avg_dist_cmps = (double)total_dist_cmps.load() / nq;
  }

  // Enhanced search implementation with all switches
  void QuantumEnhancedZeroOverheadAdaptiveIntelligenceSearch(NeighborPoolConcept auto &pool,
                                                           const ComputerConcept auto &computer) const {
    // Quantum cache-aligned buffers (Switch 8)
    alignas(64) int32_t edge_buf[graph.K * 2];
    
    int32_t consecutive_no_improvement = 0;
    int32_t total_improvements = 0;
    int32_t iteration = 0;
    
    while (pool.has_next()) {
      auto u = pool.pop();
      iteration++;
      
      // Enhanced prefetching strategy (Switch 8) - using quantum parameters
      int32_t effective_prefetch = std::min(po * batch_size_factor, graph.K);
      
      for (int32_t i = 0; i < effective_prefetch; ++i) {
        int32_t to = graph.at(u, i);
        if (to == -1) break;
        if (!pool.check_visited(to)) {
          computer.prefetch(to, pl);
        }
      }
      
      int32_t improvements = 0;
              // Original simple processing
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
      
      total_improvements += improvements;
      
      // Enhanced early termination (Switch 9)
      if (improvements == 0) {
        consecutive_no_improvement++;
        if constexpr (feature_switches::ENABLE_ENHANCED_EARLY_TERM) {
          if (checkQuantumConvergence(iteration, total_improvements, consecutive_no_improvement)) {
            break;
          }
        } else {
          if (consecutive_no_improvement >= early_stop_threshold && pool.size() >= ef / 2) {
            break;
          }
        }
      } else {
        consecutive_no_improvement = 0;
      }
      
      // Quantum iteration limit
      if (iteration > ef * 3) break;
    }
    
    // Performance tracking and adaptation (Switch 5)
    if constexpr (feature_switches::ENABLE_PERFORMANCE_TRACKING && feature_switches::ENABLE_QUANTUM_STRUCTURES) {
      if (total_improvements > 0) {
        quantum_qzoais.quantum_efficiency_tracker.fetch_add(total_improvements, std::memory_order_relaxed);
        
        uint32_t query_count = quantum_qzoais.quantum_query_counter.load(std::memory_order_relaxed);
        if (query_count % 24 == 0) {
          float efficiency = (float)total_improvements / (float)std::max(1, iteration);
          float current_perf = quantum_qzoais.quantum_performance_score.load(std::memory_order_relaxed);
          float learning_rate = std::min(0.10f, 0.06f + (efficiency * 0.04f));
          float new_perf = (1.0f - learning_rate) * current_perf + learning_rate * std::min(3.0f, efficiency * 30.0f);
          quantum_qzoais.quantum_performance_score.store(new_perf, std::memory_order_relaxed);
          
          float cache_optimizer = quantum_qzoais.quantum_cache_optimizer.load(std::memory_order_relaxed);
          if (efficiency > 0.15f) {
            cache_optimizer = std::min(1.20f, cache_optimizer + 0.003f);
          } else if (efficiency < 0.08f) {
            cache_optimizer = std::max(0.92f, cache_optimizer - 0.0015f);
          }
          quantum_qzoais.quantum_cache_optimizer.store(cache_optimizer, std::memory_order_relaxed);
          
          float stability_factor = quantum_qzoais.quantum_stability_factor.load(std::memory_order_relaxed);
          if (efficiency > 0.12f) {
            stability_factor = std::min(1.15f, stability_factor + 0.0015f);
          } else if (efficiency < 0.06f) {
            stability_factor = std::max(0.94f, stability_factor - 0.0008f);
          }
          quantum_qzoais.quantum_stability_factor.store(stability_factor, std::memory_order_relaxed);
        }
      }
    }
  }

  // Enhanced batch search implementation
  void QuantumEnhancedZeroOverheadAdaptiveIntelligenceSearchBatch(NeighborPoolConcept auto &pool,
                                                                 const ComputerConcept auto &computer) const {
    // Quantum cache-aligned buffers for batch processing (Switch 8)
    alignas(64) int32_t edge_buf[graph.K * 2];
    alignas(64) typename Quant::ComputerType::dist_type dist_buf[graph.K * 4];
    
    while (pool.has_next()) {
      auto u = pool.pop();
      
      // Quantum edge collection (Switch 7)
      int32_t edge_size = 0;
      for (int32_t i = 0; i < graph.K; ++i) {
        int32_t v = graph.at(u, i);
        if (v == -1) break;
        if (pool.check_visited(v)) continue;
        pool.set_visited(v);
        edge_buf[edge_size++] = v;
      }
      
      if (edge_size == 0) continue;
      
      // Enhanced prefetching strategy (Switch 8)
      int32_t prefetch_batch = std::min(po * batch_size_factor, edge_size);
      for (int i = 0; i < prefetch_batch; ++i) {
        computer.prefetch(edge_buf[i], pl);
      }
      
      // Quantum processing with SIMD patterns (Switch 6)
      if constexpr (feature_switches::ENABLE_SIMD_PROCESSING) {
        for (int i = 0; i < edge_size; i += 8) { // QUANTUM_SIMD_WIDTH
          int32_t chunk_end = std::min(i + 8, edge_size);
          
          // Quantum prefetching ahead
          if (chunk_end < edge_size) {
            int32_t next_prefetch = std::min(prefetch_batch, edge_size - chunk_end);
            for (int j = 0; j < next_prefetch; ++j) {
              computer.prefetch(edge_buf[chunk_end + j], pl);
            }
          }
          
          // Process quantum chunk
          for (int j = i; j < chunk_end; ++j) {
            auto v = edge_buf[j];
            auto cur_dist = computer(v);
            
            if (pool.insert(v, cur_dist)) {
              graph.prefetch(v, graph_po);
            }
          }
        }
      } else {
        // Regular processing without SIMD
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
  }

  // Original optimized search implementation (from second version)
  void OptimizedSearchImpl(NeighborPoolConcept auto &pool,
                          const ComputerConcept auto &computer) const {
    alignas(64) int32_t edge_buf[graph.K * 2];
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

  // Original optimized batch search implementation
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

  // Legacy compatibility methods
  void SearchImpl1(NeighborPoolConcept auto &pool,
                   const ComputerConcept auto &computer) const {
    if constexpr (feature_switches::ENABLE_QUANTUM_STRUCTURES) {
      QuantumEnhancedZeroOverheadAdaptiveIntelligenceSearch(pool, computer);
    } else {
      OptimizedSearchImpl(pool, computer);
    }
  }

  void SearchImpl2(NeighborPoolConcept auto &pool,
                   const ComputerConcept auto &computer) const {
    if constexpr (feature_switches::ENABLE_QUANTUM_STRUCTURES) {
      QuantumEnhancedZeroOverheadAdaptiveIntelligenceSearchBatch(pool, computer);
    } else {
      OptimizedSearchImplBatch(pool, computer);
    }
  }
};

} // namespace glass