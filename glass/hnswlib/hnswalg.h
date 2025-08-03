#pragma once

#include "hnsw_params.h"
#include "hnsw_core.h"
#include "hnsw_search.h"
#include "hnsw_construction.h"

namespace glass {

// Add methods to the existing HierarchicalNSW class defined in hnsw_core.h
template <SymComputerConcept ComputerType>
void HierarchicalNSW<ComputerType>::zeroOverheadRefreshCandidates() {
  HNSWSearchMethods<ComputerType>::zeroOverheadRefreshCandidates(*this);
}

template <SymComputerConcept ComputerType>
void HierarchicalNSW<ComputerType>::updateZeroOverheadPrefetchParams(int32_t neighbor_count, int layer, float target_recall) {
  HNSWSearchMethods<ComputerType>::updateZeroOverheadPrefetchParams(*this, neighbor_count, layer, target_recall);
}

template <SymComputerConcept ComputerType>
float HierarchicalNSW<ComputerType>::calculateZeroOverheadConfidence(size_t total_explored, size_t ef_target, size_t improvements) {
  return HNSWSearchMethods<ComputerType>::calculateZeroOverheadConfidence(*this, total_explored, ef_target, improvements);
}

template <SymComputerConcept ComputerType>
std::priority_queue<typename HierarchicalNSW<ComputerType>::pair_t> 
HierarchicalNSW<ComputerType>::ultraStreamlinedAdaptiveSearch(tableint ep_id, int32_t u,
                                                              int layer, size_t ef_search, 
                                                              float target_recall) {
  return HNSWSearchMethods<ComputerType>::ultraStreamlinedAdaptiveSearch(*this, ep_id, u, layer, ef_search, target_recall);
}

template <SymComputerConcept ComputerType>
std::priority_queue<typename HierarchicalNSW<ComputerType>::pair_t> 
HierarchicalNSW<ComputerType>::searchBaseLayer(tableint ep_id, int32_t u, int layer) {
  return HNSWSearchMethods<ComputerType>::searchBaseLayer(*this, ep_id, u, layer);
}

template <SymComputerConcept ComputerType>
std::priority_queue<typename HierarchicalNSW<ComputerType>::pair_t> 
HierarchicalNSW<ComputerType>::ultraStreamlinedMultiPathSearch(int32_t u, size_t ef_search) {
  return HNSWSearchMethods<ComputerType>::ultraStreamlinedMultiPathSearch(*this, u, ef_search);
}

template <SymComputerConcept ComputerType>
std::priority_queue<typename HierarchicalNSW<ComputerType>::pair_t> 
HierarchicalNSW<ComputerType>::searchKnn(int32_t u, size_t k) {
  return HNSWSearchMethods<ComputerType>::searchKnn(*this, u, k);
}

template <SymComputerConcept ComputerType>
void HierarchicalNSW<ComputerType>::getNeighborsByHeuristic2(std::priority_queue<pair_t> &top_candidates,
                                                              const size_t M) {
  HNSWConstructionMethods<ComputerType>::getNeighborsByHeuristic2(*this, top_candidates, M);
}

template <SymComputerConcept ComputerType>
tableint HierarchicalNSW<ComputerType>::mutuallyConnectNewElement(tableint u, std::priority_queue<pair_t> &top_candidates, int level) {
  return HNSWConstructionMethods<ComputerType>::mutuallyConnectNewElement(*this, u, top_candidates, level);
}

template <SymComputerConcept ComputerType>
void HierarchicalNSW<ComputerType>::addPoint(int32_t u, int level) {
  HNSWConstructionMethods<ComputerType>::addPoint(*this, u, level);
}

} // namespace glass