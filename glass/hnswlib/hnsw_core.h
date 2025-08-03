#pragma once

#include "glass/memory.hpp"
#include "glass/neighbor.hpp"
#include "glass/quant/computer.hpp"
#include "hnsw_params.h"
#include <atomic>
#include <cstring>
#include <mutex>
#include <queue>
#include <random>
#include <thread>
#include <algorithm>
#include <unordered_set>
#include <cmath>

namespace glass {

typedef unsigned int tableint;
typedef unsigned int linklistsizeint;

template <SymComputerConcept ComputerType> 
class HierarchicalNSW {
public:
  using dist_t = typename ComputerType::dist_type;
  using pair_t = std::pair<dist_t, tableint>;
  ComputerType computer;

  size_t max_elements_{0};
  size_t size_data_per_element_{0};
  size_t size_links_per_element_{0};
  size_t M_{0};
  size_t maxM_{0};
  size_t maxM0_{0};
  size_t ef_construction_{0};
  size_t ef_{0};

  double mult_{0.0}, revSize_{0.0};
  int maxlevel_{0};

  std::mutex global;
  std::vector<std::mutex> link_list_locks_;

  tableint enterpoint_node_{0};
  std::vector<tableint> strategic_entrypoints_;

  size_t size_links_level0_{0};

  char *data_level0_memory_{nullptr};
  char **linkLists_{nullptr};
  std::vector<int> element_levels_;
  
  // Ultra-streamlined performance metrics with zero overhead
  std::vector<std::atomic<float>> node_efficiency_scores_;
  
  // Ultra-streamlined dual-metric system for maximum efficiency
  UltraStreamlinedMetrics metrics_;

  void *dist_func_param_{nullptr};

  std::default_random_engine level_generator_;

  // Zero-overhead prefetch parameters with optimal balance
  ZeroOverheadPrefetchParams prefetch_params_;

  // Ultra-streamlined search parameters with optimal scaling
  UltraStreamlinedSearchParams search_params_;

  // Zero-overhead quality candidate management with circular buffer
  std::vector<tableint> zero_overhead_candidates_;
  std::atomic<size_t> candidate_index_{0};
  std::atomic<size_t> refresh_counter_{0};

  HierarchicalNSW(const ComputerType &computer, size_t max_elements,
                  size_t M = 16, size_t ef_construction = 200,
                  size_t random_seed = 100)
      : computer(computer), link_list_locks_(max_elements),
        element_levels_(max_elements), node_efficiency_scores_(max_elements) {
    max_elements_ = max_elements;
    M_ = M;
    maxM_ = M_;
    maxM0_ = M_ * 2;
    ef_construction_ = std::max(ef_construction, M_);
    ef_ = 10;

    level_generator_.seed(random_seed);
    strategic_entrypoints_.reserve(search_params_.MAX_SEARCH_PATHS);
    zero_overhead_candidates_.resize(16);

    // Initialize efficiency scores with ultra-streamlined baseline
    for (size_t i = 0; i < max_elements; i++) {
      node_efficiency_scores_[i].store(0.94f);
    }

    size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
    size_data_per_element_ = size_links_level0_;

    data_level0_memory_ =
        (char *)align_alloc(max_elements_ * size_data_per_element_);
    memset(data_level0_memory_, 0, max_elements * size_data_per_element_);

    enterpoint_node_ = -1;
    maxlevel_ = -1;

    linkLists_ = (char **)align_alloc(sizeof(void *) * max_elements_);
    size_links_per_element_ =
        maxM_ * sizeof(tableint) + sizeof(linklistsizeint);
    mult_ = 1 / log(1.0 * M_);
    revSize_ = 1.0 / mult_;
  }

  ~HierarchicalNSW() {
    free(data_level0_memory_);
    for (tableint i = 0; i < max_elements_; i++) {
      if (element_levels_[i] > 0)
        free(linkLists_[i]);
    }
    free(linkLists_);
  }

  int getRandomLevel(double reverse_size) {
    std::uniform_real_distribution<double> distribution(0.0, 1.0);
    double r = -log(distribution(level_generator_)) * reverse_size;
    return (int)r;
  }

  linklistsizeint *get_linklist0(tableint internal_id) const {
    return (linklistsizeint *)(data_level0_memory_ +
                               internal_id * size_data_per_element_);
  }

  linklistsizeint *get_linklist(tableint internal_id, int level) const {
    return (linklistsizeint *)(linkLists_[internal_id] +
                               (level - 1) * size_links_per_element_);
  }

  unsigned short int getListCount(linklistsizeint *ptr) const {
    return *((unsigned short int *)ptr);
  }

  void setListCount(linklistsizeint *ptr, unsigned short int size) const {
    *((unsigned short int *)(ptr)) = *((unsigned short int *)&size);
  }

  // Method declarations - implementations are in hnswalg.h
  void zeroOverheadRefreshCandidates();
  void updateZeroOverheadPrefetchParams(int32_t neighbor_count, int layer, float target_recall);
  float calculateZeroOverheadConfidence(size_t total_explored, size_t ef_target, size_t improvements);
  std::priority_queue<pair_t> ultraStreamlinedAdaptiveSearch(tableint ep_id, int32_t u,
                                                             int layer, size_t ef_search, 
                                                             float target_recall = 0.85f);
  std::priority_queue<pair_t> searchBaseLayer(tableint ep_id, int32_t u, int layer);
  std::priority_queue<pair_t> ultraStreamlinedMultiPathSearch(int32_t u, size_t ef_search);
  std::priority_queue<pair_t> searchKnn(int32_t u, size_t k);
  void getNeighborsByHeuristic2(std::priority_queue<pair_t> &top_candidates, const size_t M);
  tableint mutuallyConnectNewElement(tableint u, std::priority_queue<pair_t> &top_candidates, int level);
  void addPoint(int32_t u, int level = -1);
};

} // namespace glass