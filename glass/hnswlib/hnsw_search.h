#pragma once

#include "hnsw_core.h"

namespace glass {

template <SymComputerConcept ComputerType>
class HierarchicalNSW;

// Search-related methods for HierarchicalNSW
template <SymComputerConcept ComputerType>
class HNSWSearchMethods {
public:
  static void zeroOverheadRefreshCandidates(HierarchicalNSW<ComputerType>& hnsw) {
    // Zero-overhead circular buffer refresh - no complex validation or sorting
    size_t refresh_count = hnsw.refresh_counter_.fetch_add(1);
    if (refresh_count % 80 != 0) return;
    
    // Direct random sampling with minimal overhead - no hierarchical sampling
    size_t filled = 0;
    size_t attempts = 0;
    while (filled < 12 && attempts < 60) {
      tableint candidate = hnsw.level_generator_() % hnsw.max_elements_;
      if (hnsw.element_levels_[candidate] >= 0 && hnsw.node_efficiency_scores_[candidate].load() > 0.89f) {
        hnsw.zero_overhead_candidates_[filled] = candidate;
        filled++;
      }
      attempts++;
    }
    
    // Fill remaining with any valid candidates
    while (filled < 12 && attempts < 100) {
      tableint candidate = hnsw.level_generator_() % hnsw.max_elements_;
      if (hnsw.element_levels_[candidate] >= 0) {
        hnsw.zero_overhead_candidates_[filled] = candidate;
        filled++;
      }
      attempts++;
    }
  }

  static void updateZeroOverheadPrefetchParams(HierarchicalNSW<ComputerType>& hnsw, 
                                              int32_t neighbor_count, int layer, float target_recall) {
    // Zero-overhead prefetch depth optimization with minimal computation
    float adaptive_performance = hnsw.metrics_.adaptive_performance.load();
    
    // Ultra-minimal depth adjustment
    int optimal_depth = hnsw.prefetch_params_.base_prefetch_depth;
    if (adaptive_performance > 2.1f) {
      optimal_depth = std::min(hnsw.prefetch_params_.max_prefetch_depth, optimal_depth + 1);
    } else if (adaptive_performance < 1.65f) {
      optimal_depth = std::max(hnsw.prefetch_params_.min_prefetch_depth, optimal_depth - 1);
    }
    
    // Ultra-streamlined scaling for critical recall range
    float scaling_factor = 1.0f;
    if (target_recall > hnsw.search_params_.critical_recall_threshold) {
      float recall_excess = target_recall - hnsw.search_params_.critical_recall_threshold;
      scaling_factor = std::min(hnsw.search_params_.max_scaling_factor, 
                               hnsw.search_params_.optimal_scaling_base + recall_excess * 14.0f);
      scaling_factor = std::min(scaling_factor, hnsw.search_params_.resource_amplification_limit);
    }
    
    if (layer == 0) {
      hnsw.prefetch_params_.current_prefetch_depth = std::min(hnsw.prefetch_params_.max_prefetch_depth,
        std::max(hnsw.prefetch_params_.min_prefetch_depth, 
                (int)(optimal_depth * scaling_factor * 0.85f)));
      
      // Zero-overhead batch sizing
      float density_factor = std::min(1.6f, (float)neighbor_count / (float)hnsw.M_);
      hnsw.prefetch_params_.adaptive_batch_size = std::min(hnsw.prefetch_params_.max_batch_size,
        std::max(hnsw.prefetch_params_.min_batch_size, 
                (int)(hnsw.prefetch_params_.min_batch_size * density_factor * 0.72f)));
      
      // Minimal threshold adjustment
      hnsw.prefetch_params_.dynamic_threshold = std::max(0.34f, 
        std::min(0.74f, 0.66f - (adaptive_performance - 1.0f) * 0.12f));
    } else {
      hnsw.prefetch_params_.current_prefetch_depth = std::min(36, optimal_depth);
      hnsw.prefetch_params_.adaptive_batch_size = hnsw.prefetch_params_.min_batch_size;
    }
  }

  static float calculateZeroOverheadConfidence(HierarchicalNSW<ComputerType>& hnsw,
                                              size_t total_explored, size_t ef_target, size_t improvements) {
    if (total_explored == 0) return 0.0f;
    
    // Zero-overhead confidence calculation with minimal operations
    float exploration_ratio = std::min(1.0f, (float)total_explored / (float)ef_target);
    float improvement_rate = (float)improvements / (float)total_explored;
    
    return (0.52f * exploration_ratio + 0.48f * improvement_rate) * 
           std::min(1.58f, hnsw.metrics_.adaptive_performance.load());
  }

  static std::priority_queue<typename HierarchicalNSW<ComputerType>::pair_t> 
  ultraStreamlinedAdaptiveSearch(HierarchicalNSW<ComputerType>& hnsw,
                                tableint ep_id, int32_t u,
                                int layer, size_t ef_search, 
                                float target_recall = 0.85f) {
    // Ultra-streamlined ef scaling with minimal overhead
    size_t dynamic_ef = ef_search;
    if (target_recall > hnsw.search_params_.critical_recall_threshold) {
      float recall_excess = target_recall - hnsw.search_params_.critical_recall_threshold;
      float scaling = 1.0f + recall_excess * 14.5f * std::min(1.78f, hnsw.metrics_.adaptive_performance.load());
      scaling = std::min(scaling, hnsw.search_params_.resource_amplification_limit);
      dynamic_ef = (size_t)(ef_search * scaling);
    }
    
    // Ultra-streamlined exploration limit
    dynamic_ef = std::min(dynamic_ef, ef_search * hnsw.search_params_.exploration_limit_factor);
    
    LinearPool<typename ComputerType::dist_type> pool(
        hnsw.max_elements_, dynamic_ef, dynamic_ef);

    pool.insert(ep_id, hnsw.computer(u, ep_id));
    pool.set_visited(ep_id);

    std::vector<tableint> batch_candidates;
    batch_candidates.reserve(hnsw.prefetch_params_.max_batch_size);
    
    size_t total_explored = 0;
    size_t improvement_count = 0;
    typename ComputerType::dist_type best_distance = std::numeric_limits<typename ComputerType::dist_type>::max();

    while (pool.has_next()) {
      auto x = pool.pop();
      std::unique_lock<std::mutex> lock(hnsw.link_list_locks_[x]);
      
      int *data;
      if (layer == 0) {
        data = (int *)hnsw.get_linklist0(x);
      } else {
        data = (int *)hnsw.get_linklist(x, layer);
      }
      
      int32_t size = hnsw.getListCount((linklistsizeint *)data);
      tableint *datal = (tableint *)(data + 1);

      updateZeroOverheadPrefetchParams(hnsw, size, layer, target_recall);
      total_explored++;

      // Zero-overhead multi-level prefetching with minimal cache management
      int prefetch_depth = std::min(hnsw.prefetch_params_.current_prefetch_depth, size);
      
      // Primary prefetch level with zero overhead
      for (int32_t j = 0; j < prefetch_depth; ++j) {
        hnsw.computer.prefetch(datal[j], 3);
      }
      
      // Ultra-minimal secondary prefetching for critical recall
      if (target_recall > hnsw.search_params_.critical_recall_threshold && prefetch_depth < size) {
        int secondary_end = std::min(size, prefetch_depth + (prefetch_depth >> 4));
        for (int32_t j = prefetch_depth; j < secondary_end; ++j) {
          hnsw.computer.prefetch(datal[j], 2);
        }
      }

      // Ultra-streamlined neighbor processing with zero-overhead batching
      for (int32_t j = 0; j < size; j += hnsw.prefetch_params_.adaptive_batch_size) {
        batch_candidates.clear();
        int batch_end = std::min(j + hnsw.prefetch_params_.adaptive_batch_size, size);
        
        // Zero-overhead candidate selection
        for (int32_t k = j; k < batch_end; k++) {
          tableint y = datal[k];
          if (!pool.check_visited(y)) {
            float efficiency = hnsw.node_efficiency_scores_[y].load();
            float threshold = (target_recall > hnsw.search_params_.critical_recall_threshold) ? 
                             0.32f : hnsw.prefetch_params_.dynamic_threshold;
            
            if (efficiency > threshold || batch_candidates.size() < hnsw.prefetch_params_.min_batch_size / 4) {
              batch_candidates.push_back(y);
              pool.set_visited(y);
            }
          }
        }
        
        // Ultra-streamlined batch processing
        for (tableint y : batch_candidates) {
          auto dist = hnsw.computer(u, y);
          pool.insert(y, dist);
          
          if (dist < best_distance) {
            best_distance = dist;
            improvement_count++;
            
            // Zero-overhead efficiency update
            float current_eff = hnsw.node_efficiency_scores_[y].load();
            float performance_boost = std::min(0.38f, 0.26f * hnsw.metrics_.adaptive_performance.load());
            hnsw.node_efficiency_scores_[y].store(std::min(1.0f, current_eff + performance_boost));
          }
        }
      }
      
      // Zero-overhead early termination with minimal checks
      if (total_explored > ef_search / 12 && total_explored % 7 == 0) {
        float confidence = calculateZeroOverheadConfidence(hnsw, total_explored, dynamic_ef, improvement_count);
        
        // Ultra-streamlined confidence-based termination
        if (confidence > hnsw.search_params_.base_confidence_threshold * hnsw.metrics_.adaptive_performance.load()) {
          break;
        }
        
        // Zero-overhead efficiency-based termination for critical recall
        if (target_recall > hnsw.search_params_.critical_recall_threshold) {
          float efficiency = (float)improvement_count / (float)total_explored;
          if (efficiency < hnsw.search_params_.efficiency_termination_threshold && total_explored > dynamic_ef / 6) {
            break;
          }
        }
        
        // Ultra-streamlined exploration limit
        if (total_explored > dynamic_ef * 0.82f) {
          break;
        }
      }
    }
    
    // Ultra-responsive global metrics update with zero overhead
    size_t query_count = hnsw.metrics_.query_count.fetch_add(1) + 1;
    size_t update_frequency = std::max(21, std::min(27, (int)(24 - hnsw.metrics_.adaptive_performance.load() * 1.8f)));
    if (query_count % update_frequency == 0) {
      // Update adaptive performance with ultra-responsiveness
      float performance = (float)improvement_count / (float)std::max(1, (int)total_explored);
      float current_perf = hnsw.metrics_.adaptive_performance.load();
      float learning_rate = std::min(0.20f, 0.14f + (performance * 0.06f));
      hnsw.metrics_.adaptive_performance.store((1.0f - learning_rate) * current_perf + 
                                         learning_rate * std::min(2.6f, performance * 26.0f));
    }
    
    std::priority_queue<typename HierarchicalNSW<ComputerType>::pair_t> top_candidates;
    for (int i = 0; i < pool.size(); ++i) {
      top_candidates.emplace((typename ComputerType::dist_type)pool.dist(i), pool.id(i));
    }
    return top_candidates;
  }

  static std::priority_queue<typename HierarchicalNSW<ComputerType>::pair_t> 
  searchBaseLayer(HierarchicalNSW<ComputerType>& hnsw, tableint ep_id, int32_t u, int layer) {
    return ultraStreamlinedAdaptiveSearch(hnsw, ep_id, u, layer, hnsw.ef_construction_);
  }

  static std::priority_queue<typename HierarchicalNSW<ComputerType>::pair_t> 
  ultraStreamlinedMultiPathSearch(HierarchicalNSW<ComputerType>& hnsw, int32_t u, size_t ef_search) {
    if (hnsw.enterpoint_node_ == (tableint)-1) {
      return std::priority_queue<typename HierarchicalNSW<ComputerType>::pair_t>();
    }

    // Ultra-streamlined target recall estimation
    float target_recall = std::min(0.92f, 0.79f + (float)ef_search / 340.0f);
    
    // Zero-overhead path selection
    int num_paths = hnsw.search_params_.MIN_SEARCH_PATHS;
    if (ef_search > 16) {
      num_paths = std::min(hnsw.search_params_.MAX_SEARCH_PATHS, 
                          hnsw.search_params_.MIN_SEARCH_PATHS + (int)(ef_search / 38));
    }

    // Collect entry points with ultra-streamlined selection
    std::vector<tableint> entry_points;
    entry_points.push_back(hnsw.enterpoint_node_);
    
    // Add strategic entry points
    for (size_t i = 0; i < std::min((size_t)(num_paths - 1), hnsw.strategic_entrypoints_.size()); i++) {
      if (hnsw.strategic_entrypoints_[i] != hnsw.enterpoint_node_) {
        entry_points.push_back(hnsw.strategic_entrypoints_[i]);
      }
    }

    // Generate additional entry points with zero-overhead circular buffer
    zeroOverheadRefreshCandidates(hnsw);
    
    while (entry_points.size() < num_paths && hnsw.zero_overhead_candidates_.size() > 0) {
      size_t idx = hnsw.candidate_index_.fetch_add(1) % hnsw.zero_overhead_candidates_.size();
      tableint candidate = hnsw.zero_overhead_candidates_[idx];
      
      if (candidate != 0 && hnsw.element_levels_[candidate] >= 0) {
        bool is_duplicate = false;
        for (tableint ep : entry_points) {
          if (ep == candidate) {
            is_duplicate = true;
            break;
          }
        }
        
        if (!is_duplicate) {
          float efficiency_threshold = 0.54f + hnsw.metrics_.adaptive_performance.load() * 0.12f;
          if (hnsw.node_efficiency_scores_[candidate].load() > efficiency_threshold) {
            entry_points.push_back(candidate);
          }
        }
      }
    }

    // Ultra-streamlined multi-path execution
    std::priority_queue<typename HierarchicalNSW<ComputerType>::pair_t> final_results;
    std::unordered_set<tableint> seen_nodes;
    
    for (size_t path_idx = 0; path_idx < entry_points.size(); path_idx++) {
      tableint currObj = entry_points[path_idx];
      
      // Zero-overhead path-specific ef scaling
      float path_multiplier = 1.0f + path_idx * 0.38f * std::min(1.58f, hnsw.metrics_.adaptive_performance.load());
      path_multiplier = std::min(path_multiplier, hnsw.search_params_.resource_amplification_limit);
      size_t path_ef = (size_t)(ef_search * path_multiplier);
      
      if (hnsw.element_levels_[currObj] > 0) {
        auto curdist = hnsw.computer(u, currObj);
        for (int level = hnsw.element_levels_[currObj]; level > 0; level--) {
          bool changed = true;
          while (changed) {
            changed = false;
            std::unique_lock<std::mutex> lock(hnsw.link_list_locks_[currObj]);
            unsigned int *data = hnsw.get_linklist(currObj, level);
            int size = hnsw.getListCount(data);
            tableint *datal = (tableint *)(data + 1);
            
            for (int i = 0; i < size; i++) {
              if (i + 1 < size) {
                hnsw.computer.prefetch(datal[i + 1], 1);
              }
              tableint cand = datal[i];
              auto d = hnsw.computer(u, cand);
              if (d < curdist) {
                curdist = d;
                currObj = cand;
                changed = true;
              }
            }
          }
        }
      }
      
      // Ultra-streamlined search at level 0
      auto path_results = ultraStreamlinedAdaptiveSearch(hnsw, currObj, u, 0, path_ef, target_recall);
      
      // Direct result integration
      std::vector<typename HierarchicalNSW<ComputerType>::pair_t> path_vec;
      while (!path_results.empty()) {
        auto [dist, node] = path_results.top();
        path_results.pop();
        path_vec.emplace_back(dist, node);
      }
      std::sort(path_vec.begin(), path_vec.end());
      
      // Zero-overhead unique result addition
      for (const auto& [dist, node] : path_vec) {
        if (seen_nodes.find(node) == seen_nodes.end()) {
          seen_nodes.insert(node);
          final_results.emplace(dist, node);
          if (final_results.size() >= ef_search * 2) break;
        }
      }
      if (final_results.size() >= ef_search * 2) break;
    }

    return final_results;
  }

  static std::priority_queue<typename HierarchicalNSW<ComputerType>::pair_t> 
  searchKnn(HierarchicalNSW<ComputerType>& hnsw, int32_t u, size_t k) {
    return ultraStreamlinedMultiPathSearch(hnsw, u, std::max(hnsw.ef_, k));
  }
};

} // namespace glass