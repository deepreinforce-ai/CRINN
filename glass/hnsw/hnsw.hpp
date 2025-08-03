#pragma once

#include "glass/builder.hpp"
#include "glass/common.hpp"
#include "glass/graph.hpp"
#include "glass/graph_statistic.hpp"
#include "glass/hnsw/HNSWInitializer.hpp"
#include "glass/hnswlib/hnswalg.h"
#include "glass/memory.hpp"
#include "glass/quant/quant.hpp"
#include "glass/quant/quant_base.hpp"
#include "glass/utils.hpp"
#include <chrono>
#include <memory>
#include <algorithm>
#include <random>
#include <unordered_set>
#include <cmath>

namespace glass {

template <SymComputableQuantConcept QuantType> struct HNSW : public Builder {
  int32_t R, efConstruction;
  double construction_time;

  bool ENABLE_MULTI_ENTRY_POINTS = true;

  struct MultiEntryParams {
    int32_t max_entry_points = 9;
    float distance_threshold = 0.18f;
    float diversity_weight = 1.72f;
  } entry_params_;

  HNSW(int32_t R = 32, int32_t L = 200) : R(R), efConstruction(L) {}

  void constructMultiEntryPoints(const HierarchicalNSW<typename QuantType::SymComputerType>& hnsw,
                                std::vector<int32_t>& entry_points,
                                const auto& computer, int32_t N) {
    entry_points.clear();
    entry_points.push_back(hnsw.enterpoint_node_);
    
    if (!ENABLE_MULTI_ENTRY_POINTS) {
      return;
    }
    
    std::vector<int32_t> candidates;
    for (int32_t i = 0; i < N; i++) {
      if (hnsw.element_levels_[i] > 0 && i != hnsw.enterpoint_node_) {
        candidates.push_back(i);
      }
    }
    
    if (candidates.empty()) return;
    
    while (entry_points.size() < entry_params_.max_entry_points && !candidates.empty()) {
      int32_t best_candidate = -1;
      float best_score = -1.0f;
      size_t best_idx = 0;
      
      for (size_t i = 0; i < candidates.size(); i++) {
        int32_t candidate = candidates[i];
        
        float score = (float)hnsw.element_levels_[candidate] * 3.0f;
        
        float min_dist = std::numeric_limits<float>::max();
        for (int32_t ep : entry_points) {
          float dist = computer(candidate, ep);
          min_dist = std::min(min_dist, dist);
        }
        
        if (min_dist > entry_params_.distance_threshold) {
          score += min_dist * entry_params_.diversity_weight;
        }
        
        if (hnsw.element_levels_[candidate] >= 2) {
          score *= 1.5f;
        }
        
        if (score > best_score) {
          best_score = score;
          best_candidate = candidate;
          best_idx = i;
        }
      }
      
      if (best_candidate != -1) {
        entry_points.push_back(best_candidate);
        candidates.erase(candidates.begin() + best_idx);
      } else {
        break;
      }
    }
  }

  Graph<int32_t> Build(const float *data, int32_t N, int32_t dim) override {
    QuantType quant(dim);
    quant.train(data, N);
    quant.add(data, N);
    
    HierarchicalNSW hnsw(quant.get_sym_computer(), N, R / 2, efConstruction);
    
    running_stats_printer_t printer(N, "HNSW Construction");
    
    hnsw.addPoint(0);
    
    #pragma omp parallel for schedule(dynamic)
    for (int32_t i = 1; i < N; ++i) {
      hnsw.addPoint(i);
      printer.progress += 1;
      printer.refresh();
    }
    
    auto time = std::chrono::high_resolution_clock::now();
    construction_time = std::chrono::duration<double>(time - printer.start_time).count();
    
    Graph<int32_t> graph(N, R);
    
    #pragma omp parallel for schedule(static, 48)
    for (int64_t i = 0; i < N; ++i) {
      int32_t *edges = (int32_t *)hnsw.get_linklist0(i);
      int32_t degree = edges[0];
      
      std::vector<std::pair<float, int32_t>> edge_candidates;
      edge_candidates.reserve(degree);
      
      for (int j = 1; j <= degree; ++j) {
        int32_t neighbor = edges[j];
        if (neighbor >= 0 && neighbor < N && neighbor != i) {
          float dist = quant.get_sym_computer()(i, neighbor);
          edge_candidates.emplace_back(dist, neighbor);
        }
      }
      
      if (edge_candidates.size() > R) {
        std::sort(edge_candidates.begin(), edge_candidates.end());
        edge_candidates.resize(R);
      }
      
      std::sort(edge_candidates.begin(), edge_candidates.end());
      
      for (size_t j = 0; j < std::min((size_t)R, edge_candidates.size()); ++j) {
        graph.at(i, j) = edge_candidates[j].second;
      }
      
      for (int j = edge_candidates.size(); j < R; ++j) {
        graph.at(i, j) = -1;
      }
    }
    
    auto initializer = std::make_unique<HNSWInitializer>(N, R / 2);
    initializer->ep = hnsw.enterpoint_node_;
    
    std::vector<int32_t> entry_points;
    constructMultiEntryPoints(hnsw, entry_points, quant.get_sym_computer(), N);
    
    graph.eps.clear();
    for (int32_t ep : entry_points) {
      graph.eps.push_back(ep);
    }
    
    for (int64_t i = 0; i < N; ++i) {
      int32_t level = hnsw.element_levels_[i];
      initializer->levels[i] = level;
      
      if (level > 0) {
        initializer->lists[i] = (int *)align_alloc(level * R * 2, true, -1);
        for (int32_t j = 1; j <= level; ++j) {
          int32_t *level_edges = (int32_t *)hnsw.get_linklist(i, j);
          int32_t edge_count = std::min(level_edges[0], R);
          for (int32_t k = 1; k <= edge_count; ++k) {
            initializer->at(j, i, k - 1) = level_edges[k];
          }
        }
      }
    }
    
    graph.initializer = std::move(initializer);
    print_degree_statistic(graph);
    return graph;
  }

  double GetConstructionTime() const override { return construction_time; }
};

inline std::unique_ptr<Builder>
create_hnsw(const std::string &metric, const std::string &quantizer = "BF16",
            int32_t R = 32, int32_t L = 200) {
  auto m = metric_map[metric];
  auto qua = quantizer_map[quantizer];
  
  if (qua == QuantizerType::FP32) {
    if (m == Metric::L2) {
      return std::make_unique<HNSW<FP32Quantizer<Metric::L2>>>(R, L);
    }
    if (m == Metric::IP) {
      return std::make_unique<HNSW<FP32Quantizer<Metric::IP>>>(R, L);
    }
  }
  if (qua == QuantizerType::BF16) {
    if (m == Metric::L2) {
      return std::make_unique<HNSW<BF16Quantizer<Metric::L2>>>(R, L);
    }
    if (m == Metric::IP) {
      return std::make_unique<HNSW<BF16Quantizer<Metric::IP>>>(R, L);
    }
  }
  if (qua == QuantizerType::FP16) {
    if (m == Metric::L2) {
      return std::make_unique<HNSW<FP16Quantizer<Metric::L2>>>(R, L);
    }
    if (m == Metric::IP) {
      return std::make_unique<HNSW<FP16Quantizer<Metric::IP>>>(R, L);
    }
  }
  if (qua == QuantizerType::SQ8U) {
    if (m == Metric::L2) {
      return std::make_unique<HNSW<SQ8QuantizerUniform<Metric::L2>>>(R, L);
    }
    if (m == Metric::IP) {
      return std::make_unique<HNSW<SQ8QuantizerUniform<Metric::IP>>>(R, L);
    }
  }
  if (qua == QuantizerType::SQ4U) {
    if (m == Metric::L2) {
      return std::make_unique<HNSW<SQ4QuantizerUniform<Metric::L2>>>(R, L);
    }
    if (m == Metric::IP) {
      return std::make_unique<HNSW<SQ4QuantizerUniform<Metric::IP>>>(R, L);
    }
  }
  if (qua == QuantizerType::SQ2U) {
    if (m == Metric::L2) {
      return std::make_unique<HNSW<SQ2QuantizerUniform<Metric::L2>>>(R, L);
    }
    if (m == Metric::IP) {
      return std::make_unique<HNSW<SQ2QuantizerUniform<Metric::IP>>>(R, L);
    }
  }
  if (qua == QuantizerType::SQ1) {
    if (m == Metric::IP) {
      return std::make_unique<HNSW<SQ1Quantizer<Metric::IP>>>(R, L);
    }
  }
  
  printf("Quantizer type %s not supported\n", quantizer.c_str());
  return nullptr;
}

} // namespace glass