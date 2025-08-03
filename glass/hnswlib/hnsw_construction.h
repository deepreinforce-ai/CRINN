#pragma once

#include "hnsw_core.h"
#include "hnsw_search.h"

namespace glass {

template <SymComputerConcept ComputerType>
class HierarchicalNSW;

// Construction-related methods for HierarchicalNSW
template <SymComputerConcept ComputerType>
class HNSWConstructionMethods {
public:
  static void getNeighborsByHeuristic2(HierarchicalNSW<ComputerType>& hnsw,
                                      std::priority_queue<typename HierarchicalNSW<ComputerType>::pair_t> &top_candidates,
                                      const size_t M) {
    if (top_candidates.size() < M) {
      return;
    }

    std::priority_queue<typename HierarchicalNSW<ComputerType>::pair_t> queue_closest;
    std::vector<typename HierarchicalNSW<ComputerType>::pair_t> return_list;
    while (top_candidates.size() > 0) {
      auto [dist, u] = top_candidates.top();
      top_candidates.pop();
      queue_closest.emplace(-dist, u);
    }

    while (queue_closest.size()) {
      if (return_list.size() >= M) {
        break;
      }
      auto [dist, u] = queue_closest.top();
      queue_closest.pop();
      auto dist_to_query = -dist;
      bool good = true;

      // Zero-overhead diversity threshold
      float diversity_threshold = 0.62f + 0.38f * (return_list.size() / (float)M);
      diversity_threshold *= (1.0f + std::min(0.20f, (hnsw.metrics_.adaptive_performance.load() - 1.0f) * 0.12f));
      
      for (auto [_, v] : return_list) {
        auto curdist = hnsw.computer(u, v);
        if (curdist < dist_to_query * diversity_threshold) {
          good = false;
          break;
        }
      }
      if (good) {
        return_list.push_back({dist, u});
      }
    }

    for (auto [dist, u] : return_list) {
      top_candidates.emplace(-dist, u);
    }
  }

  static tableint mutuallyConnectNewElement(HierarchicalNSW<ComputerType>& hnsw,
                                           tableint u, std::priority_queue<typename HierarchicalNSW<ComputerType>::pair_t> &top_candidates, int level) {
    size_t Mcurmax = level ? hnsw.maxM_ : hnsw.maxM0_;
    getNeighborsByHeuristic2(hnsw, top_candidates, hnsw.M_);

    std::vector<tableint> selectedNeighbors;
    selectedNeighbors.reserve(hnsw.M_);
    while (top_candidates.size() > 0) {
      selectedNeighbors.push_back(top_candidates.top().second);
      top_candidates.pop();
    }

    tableint next_closest_entry_point = selectedNeighbors.empty() ? u : selectedNeighbors.back();
    linklistsizeint *ll_cur;
    if (level == 0)
      ll_cur = hnsw.get_linklist0(u);
    else
      ll_cur = hnsw.get_linklist(u, level);

    hnsw.setListCount(ll_cur, selectedNeighbors.size());
    tableint *data = (tableint *)(ll_cur + 1);
    for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
      data[idx] = selectedNeighbors[idx];
    }

    // Zero-overhead efficiency score update
    if (level == 0 && !selectedNeighbors.empty()) {
      float base_efficiency = std::min(1.0f, (float)selectedNeighbors.size() / (float)hnsw.M_ * 0.88f);
      
      // Zero-overhead efficiency calculation
      float neighbor_efficiency_sum = 0.0f;
      for (tableint neighbor : selectedNeighbors) {
        neighbor_efficiency_sum += hnsw.node_efficiency_scores_[neighbor].load();
      }
      float avg_neighbor_efficiency = neighbor_efficiency_sum / selectedNeighbors.size();
      
      float final_efficiency = std::min(1.0f, base_efficiency + avg_neighbor_efficiency * 0.18f * 
                                        std::min(1.28f, hnsw.metrics_.adaptive_performance.load()));
      hnsw.node_efficiency_scores_[u].store(final_efficiency);
    }

    for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
      std::unique_lock<std::mutex> lock(
          hnsw.link_list_locks_[selectedNeighbors[idx]]);

      linklistsizeint *ll_other;
      if (level == 0)
        ll_other = hnsw.get_linklist0(selectedNeighbors[idx]);
      else
        ll_other = hnsw.get_linklist(selectedNeighbors[idx], level);

      size_t sz_link_list_other = hnsw.getListCount(ll_other);
      tableint *data = (tableint *)(ll_other + 1);

      if (sz_link_list_other < Mcurmax) {
        data[sz_link_list_other] = u;
        hnsw.setListCount(ll_other, sz_link_list_other + 1);
      } else {
        auto d_max = hnsw.computer(u, selectedNeighbors[idx]);
        std::priority_queue<typename HierarchicalNSW<ComputerType>::pair_t> candidates;
        candidates.emplace(d_max, u);

        for (size_t j = 0; j < sz_link_list_other; j++) {
          candidates.emplace(hnsw.computer(data[j], selectedNeighbors[idx]),
                             data[j]);
        }

        getNeighborsByHeuristic2(hnsw, candidates, Mcurmax);

        int indx = 0;
        while (candidates.size() > 0) {
          data[indx] = candidates.top().second;
          candidates.pop();
          indx++;
        }
        hnsw.setListCount(ll_other, indx);
      }
    }
    return next_closest_entry_point;
  }

  static void addPoint(HierarchicalNSW<ComputerType>& hnsw, int32_t u, int level = -1) {
    std::unique_lock<std::mutex> lock_el(hnsw.link_list_locks_[u]);
    int curlevel = hnsw.getRandomLevel(hnsw.mult_);
    if (level > 0)
      curlevel = level;

    hnsw.element_levels_[u] = curlevel;

    std::unique_lock<std::mutex> templock(hnsw.global);
    int maxlevelcopy = hnsw.maxlevel_;
    if (curlevel <= maxlevelcopy)
      templock.unlock();
    tableint currObj = hnsw.enterpoint_node_;

    if (curlevel) {
      hnsw.linkLists_[u] =
          (char *)align_alloc(hnsw.size_links_per_element_ * curlevel + 1);
      memset(hnsw.linkLists_[u], 0, hnsw.size_links_per_element_ * curlevel + 1);
    }

    if ((signed)currObj != -1) {
      if (curlevel < maxlevelcopy) {
        auto curdist = hnsw.computer(u, currObj);
        for (int level = maxlevelcopy; level > curlevel; level--) {
          bool changed = true;
          while (changed) {
            changed = false;
            unsigned int *data;
            std::unique_lock<std::mutex> lock(hnsw.link_list_locks_[currObj]);
            data = hnsw.get_linklist(currObj, level);
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

      for (int level = std::min(curlevel, maxlevelcopy); level >= 0; level--) {
        std::priority_queue<typename HierarchicalNSW<ComputerType>::pair_t> top_candidates =
            HNSWSearchMethods<ComputerType>::searchBaseLayer(hnsw, currObj, u, level);
        currObj = mutuallyConnectNewElement(hnsw, u, top_candidates, level);
      }
    } else {
      hnsw.enterpoint_node_ = 0;
      hnsw.maxlevel_ = curlevel;
    }

    if (curlevel > maxlevelcopy) {
      hnsw.enterpoint_node_ = u;
      hnsw.maxlevel_ = curlevel;
    }

    // Zero-overhead entry point management
    if (curlevel >= maxlevelcopy - 1) {
      if (hnsw.strategic_entrypoints_.size() < hnsw.search_params_.MAX_SEARCH_PATHS) {
        hnsw.strategic_entrypoints_.push_back(u);
      } else {
        // Replace entry point with lowest efficiency score
        auto min_it = std::min_element(hnsw.strategic_entrypoints_.begin(), 
                                      hnsw.strategic_entrypoints_.end(),
                                      [&hnsw](tableint a, tableint b) {
                                        return hnsw.node_efficiency_scores_[a].load() < hnsw.node_efficiency_scores_[b].load();
                                      });
        
        float performance_threshold = hnsw.metrics_.adaptive_performance.load() * 0.10f;
        if (hnsw.node_efficiency_scores_[u].load() > hnsw.node_efficiency_scores_[*min_it].load() + performance_threshold) {
          *min_it = u;
        }
      }
    }
  }
};

} // namespace glass