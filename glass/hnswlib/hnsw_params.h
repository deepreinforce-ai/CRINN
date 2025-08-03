#pragma once

#include <atomic>
#include <vector>

namespace glass {

// Zero-overhead prefetch parameters with optimal balance
struct ZeroOverheadPrefetchParams {
  int32_t base_prefetch_depth = 36;
  int32_t min_prefetch_depth = 24;
  int32_t max_prefetch_depth = 48;
  int32_t current_prefetch_depth = 36;
  int32_t adaptive_batch_size = 36;
  int32_t min_batch_size = 24;
  int32_t max_batch_size = 54;
  float dynamic_threshold = 0.66f;
};

// Ultra-streamlined search parameters with optimal scaling
struct UltraStreamlinedSearchParams {
  static constexpr int MAX_SEARCH_PATHS = 3;
  static constexpr int MIN_SEARCH_PATHS = 2;
  float base_confidence_threshold = 0.62f;
  float critical_recall_threshold = 0.81f;
  float optimal_scaling_base = 5.4f;
  float max_scaling_factor = 8.8f;
  size_t exploration_limit_factor = 4;
  float efficiency_termination_threshold = 0.036f;
  float resource_amplification_limit = 3.8f;
};

// Ultra-streamlined dual-metric system for maximum efficiency
struct UltraStreamlinedMetrics {
  std::atomic<float> adaptive_performance{1.88f};
  std::atomic<size_t> query_count{0};
};

} // namespace glass