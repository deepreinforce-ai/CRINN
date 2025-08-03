#pragma once

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <vector>
#include <algorithm>
#include <random>
#include <atomic>

#include "glass/hnsw/HNSWInitializer.hpp"
#include "glass/memory.hpp"
#include "glass/neighbor.hpp"
#include "glass/quant/computer.hpp"
#include "glass/simd/prefetch.hpp"

namespace glass {

template <typename GraphType>
concept GraphConcept = requires(GraphType graph, int32_t i, int32_t j) {
  graph.size();
  graph.range();
  graph.at(i, j);
  graph.edges(i);
  graph.degree(i);
};

template <typename node_t> struct Graph {
  int32_t N;
  int32_t K;

  constexpr static int EMPTY_ID = -1;

  node_t *data = nullptr;

  std::unique_ptr<HNSWInitializer> initializer = nullptr;

  std::vector<int> eps;
  
  // Ultra-compact 8-bit signatures for maximum cache efficiency
  alignas(64) std::vector<uint8_t> quality_signatures; // [quality:8] ultra-compact
  alignas(64) std::vector<int32_t> elite_paths; // Cache-line aligned strategic paths
  
  // Zero-overhead parameters optimized for branch prediction
  struct UltraParams {
    uint8_t quality_threshold = 0xC0;    // Top 25% (192/255)
    uint8_t diversity_threshold = 0x40;  // Minimum difference for diversity
    int32_t max_paths = 3;               // Optimal path count
    int32_t prefetch_window = 2;         // Cache-optimal window
  } params;

  // Minimal atomic state for zero-overhead adaptation
  mutable std::atomic<uint32_t> search_cycles{0};

  Graph() = default;

  Graph(node_t *edges, int32_t N, int32_t K) : N(N), K(K), data(edges) {
    if (N > 0) init_ultra_compact();
  }

  Graph(int32_t N, int32_t K)
      : N(N), K(K),
        data((node_t *)align_alloc((size_t)N * K * sizeof(node_t), true, -1)) {
    if (N > 0) init_ultra_compact();
  }

  Graph(const Graph &g) = delete;
  Graph(Graph &&g) { swap(*this, g); }
  Graph &operator=(const Graph &rhs) = delete;
  Graph &operator=(Graph &&rhs) {
    swap(*this, rhs);
    return *this;
  }

  friend void swap(Graph &lhs, Graph &rhs) {
    using std::swap;
    swap(lhs.N, rhs.N);
    swap(lhs.K, rhs.K);
    swap(lhs.data, rhs.data);
    swap(lhs.initializer, rhs.initializer);
    swap(lhs.eps, rhs.eps);
    swap(lhs.quality_signatures, rhs.quality_signatures);
    swap(lhs.elite_paths, rhs.elite_paths);
    swap(lhs.params, rhs.params);
  }

  void init_ultra_compact() {
    quality_signatures.resize(N, 0x80); // Default: medium quality
    elite_paths.reserve(params.max_paths);
  }

  void init(int32_t N, int K) {
    data = (node_t *)align_alloc((size_t)N * K * sizeof(node_t), true, -1);
    this->K = K;
    this->N = N;
    init_ultra_compact();
  }

  ~Graph() { free(data); }

  int32_t size() const { return N; }
  int32_t range() const { return K; }
  const int *edges(int32_t u) const { return data + (int64_t)K * u; }
  int *edges(int32_t u) { return data + (int64_t)K * u; }
  node_t at(int32_t i, int32_t j) const { return data[(int64_t)i * K + j]; }
  node_t &at(int32_t i, int32_t j) { return data[(int64_t)i * K + j]; }

  int32_t degree(int32_t i) const {
    int32_t deg = 0;
    while (deg < range() && at(i, deg) != EMPTY_ID) {
      ++deg;
    }
    return deg;
  }

  void prefetch(int32_t u, int32_t lines) const {
    mem_prefetch((char *)edges(u), lines);
  }

  // Ultra-fast 8-bit quality signature computation with single-pass optimization
  void compute_ultra_compact_signatures() {
    if (N == 0) return;
    
    // Cache-friendly single-pass computation
    for (int32_t i = 0; i < N; ++i) {
      int32_t deg = degree(i);
      
      // Ultra-compact quality encoding: base score from degree ratio
      uint8_t base_quality = static_cast<uint8_t>((deg * 255) / K);
      
      // Neighbor quality boost using only first 2 neighbors for cache efficiency
      uint8_t neighbor_boost = 0;
      if (deg >= 2) {
        int32_t n1 = at(i, 0);
        int32_t n2 = at(i, 1);
        if (n1 != EMPTY_ID && n2 != EMPTY_ID && n1 < N && n2 < N) {
          int32_t combined_degree = degree(n1) + degree(n2);
          neighbor_boost = static_cast<uint8_t>(std::min(63, combined_degree * 31 / K));
        }
      }
      
      // Single-instruction quality fusion
      quality_signatures[i] = std::min(255, static_cast<int>(base_quality) + neighbor_boost);
    }
  }

  // Ultra-fast elite path selection using SIMD-friendly operations
  void select_ultra_elite_paths() {
    if (eps.empty()) return;
    
    elite_paths.clear();
    elite_paths.push_back(eps[0]); // Primary path
    
    // Ultra-fast quality-based candidate collection
    alignas(64) std::vector<int32_t> candidates;
    candidates.reserve(N / 16); // Conservative cache-friendly reserve
    
    // SIMD-friendly loop for quality filtering
    for (int32_t i = 0; i < N; ++i) {
      if (__builtin_expect(quality_signatures[i] >= params.quality_threshold, 0)) {
        candidates.push_back(i);
      }
    }
    
    // Ultra-fast diversity selection using bit operations
    for (int32_t candidate : candidates) {
      if (__builtin_expect(elite_paths.size() >= params.max_paths, 0)) break;
      
      // Lightning-fast diversity check using quality signature differences
      bool is_diverse = true;
      uint8_t candidate_sig = quality_signatures[candidate];
      
      for (int32_t existing : elite_paths) {
        if (__builtin_expect(candidate == existing, 0)) {
          is_diverse = false;
          break;
        }
        
        // Single-instruction diversity test
        uint8_t sig_diff = candidate_sig > quality_signatures[existing] ? 
                          candidate_sig - quality_signatures[existing] : 
                          quality_signatures[existing] - candidate_sig;
        
        if (__builtin_expect(sig_diff < params.diversity_threshold, 1)) {
          is_diverse = false;
          break;
        }
      }
      
      if (__builtin_expect(is_diverse, 1)) {
        elite_paths.push_back(candidate);
      }
    }
  }

  // Zero-overhead intelligent prefetching with cache-line optimization
  GLASS_INLINE void ultra_prefetch(int32_t u) const {
    if (__builtin_expect(u < 0 || u >= N, 0)) return;
    
    // Always prefetch current node with cache-line alignment
    prefetch(u, 1);
    
    // Quality-gated neighbor prefetching using 8-bit comparison
    uint8_t node_quality = quality_signatures[u];
    if (__builtin_expect(node_quality >= 0xA0, 1)) { // Top 37.5%
      int32_t deg = degree(u);
      int32_t prefetch_count = std::min(params.prefetch_window, deg);
      
      // Cache-optimized prefetch loop
      for (int32_t i = 0; i < prefetch_count; ++i) {
        int32_t neighbor = at(u, i);
        if (__builtin_expect(neighbor != EMPTY_ID && neighbor < N, 1)) {
          // Only prefetch high-quality neighbors for cache efficiency
          if (__builtin_expect(quality_signatures[neighbor] >= 0xB0, 1)) {
            prefetch(neighbor, 1);
          }
        }
      }
    }
  }

  // Ultra-streamlined search initialization with zero overhead
  void initialize_search(NeighborPoolConcept auto &pool,
                         const ComputerConcept auto &computer) const {
    if (__builtin_expect(initializer != nullptr, 0)) {
      initializer->initialize(pool, computer);
    } else {
      // Use elite paths with zero-overhead iteration
      const auto& paths = elite_paths.empty() ? eps : elite_paths;
      
      for (int32_t entry : paths) {
        if (__builtin_expect(entry >= 0 && entry < N, 1)) {
          pool.insert(entry, computer(entry));
          pool.set_visited(entry);
          ultra_prefetch(entry);
        }
      }
    }
  }

  // Minimal overhead performance tracking with atomic efficiency
  void update_performance_metrics(float efficiency) const {
    // Power-of-2 modulo for ultra-fast adaptation check
    uint32_t cycle = search_cycles.fetch_add(1) + 1;
    if (__builtin_expect((cycle & 127) == 0, 0)) { // Every 128 searches
      // Ultra-simple adaptive tuning
      if (efficiency < 0.6f && params.quality_threshold > 0x90) {
        const_cast<Graph*>(this)->params.quality_threshold -= 0x10;
      } else if (efficiency > 0.9f && params.quality_threshold < 0xE0) {
        const_cast<Graph*>(this)->params.quality_threshold += 0x10;
      }
    }
  }

  void save(const std::string &filename) const {
    static_assert(std::is_same_v<node_t, int32_t>);
    std::ofstream writer(filename.c_str(), std::ios::binary);
    int nep = eps.size();
    writer.write((char *)&nep, 4);
    writer.write((char *)eps.data(), nep * 4);
    writer.write((char *)&N, 4);
    writer.write((char *)&K, 4);
    writer.write((char *)data, (int64_t)N * K * 4);
    
    // Save ultra-compact structures
    if (!quality_signatures.empty()) {
      writer.write((char *)quality_signatures.data(), N);
    }
    
    int n_elite = elite_paths.size();
    writer.write((char *)&n_elite, 4);
    if (n_elite > 0) {
      writer.write((char *)elite_paths.data(), n_elite * 4);
    }
    
    if (initializer) {
      initializer->save(writer);
    }
    printf("Ultra-Efficient Graph Saving done\n");
  }

  void load(const std::string &filename, const std::string &format) {
    if (format == "glass") {
      load(filename);
    } else if (format == "diskann") {
      load_diskann(filename);
    } else if (format == "hnswlib") {
      load_hnswlib(filename);
    } else {
      printf("Unknown graph format\n");
      exit(-1);
    }
    
    finalize_ultra_compact();
  }

  void load(const std::string &filename) {
    static_assert(std::is_same_v<node_t, int32_t>);
    free(data);
    std::ifstream reader(filename.c_str(), std::ios::binary);
    int nep;
    reader.read((char *)&nep, 4);
    eps.resize(nep);
    reader.read((char *)eps.data(), nep * 4);
    reader.read((char *)&N, 4);
    reader.read((char *)&K, 4);
    data = (node_t *)align_alloc((int64_t)N * K * 4, true, -1);
    reader.read((char *)data, N * K * 4);
    
    init_ultra_compact();
    
    // Load ultra-compact structures if available
    if (reader.peek() != EOF && quality_signatures.size() == N) {
      reader.read((char *)quality_signatures.data(), N);
    }
    
    if (reader.peek() != EOF) {
      int n_elite;
      reader.read((char *)&n_elite, 4);
      if (n_elite > 0) {
        elite_paths.resize(n_elite);
        reader.read((char *)elite_paths.data(), n_elite * 4);
      }
    }
    
    if (reader.peek() != EOF) {
      initializer = std::make_unique<HNSWInitializer>(N);
      initializer->load(reader);
    }
    
    finalize_ultra_compact();
    printf("Ultra-Efficient Graph Loading done\n");
  }

  void finalize_ultra_compact() {
    // Only compute if needed
    bool needs_computation = quality_signatures.empty() || 
                           std::all_of(quality_signatures.begin(), quality_signatures.end(), 
                                     [](uint8_t sig) { return sig == 0x80; });
    
    if (needs_computation) {
      compute_ultra_compact_signatures();
    }
    
    if (elite_paths.empty()) {
      select_ultra_elite_paths();
    }
  }

  void load_diskann(const std::string &filename) {
    static_assert(std::is_same_v<node_t, int32_t>);
    free(data);
    std::ifstream reader(filename.c_str(), std::ios::binary);
    size_t size;
    reader.read((char *)&size, 8);
    reader.read((char *)&K, 4);
    eps.resize(1);

    reader.read((char *)&eps[0], 4);
    size_t x;
    reader.read((char *)&x, 8);
    N = 0;
    while (reader.tellg() < size) {
      N++;
      int32_t cur_k;
      reader.read((char *)&cur_k, 4);
      reader.seekg(cur_k * 4, reader.cur);
    }
    reader.seekg(24, reader.beg);
    data = (node_t *)align_alloc((int64_t)N * K * 4, true, -1);
    memset(data, -1, (int64_t)N * K * 4);
    for (int i = 0; i < N; ++i) {
      int cur_k;
      reader.read((char *)&cur_k, 4);
      reader.read((char *)edges(i), 4 * cur_k);
    }
    
    init_ultra_compact();
    finalize_ultra_compact();
  }

  void load_hnswlib(const std::string &filename) {
    static_assert(std::is_same_v<node_t, int32_t>);
    free(data);
    std::ifstream reader(filename.c_str(), std::ios::binary);
    reader.seekg(8, std::ios::cur);
    size_t max_elements;
    reader.read((char *)&max_elements, 8);
    N = max_elements;

    reader.seekg(8, std::ios::cur);
    size_t size_per_element;
    reader.read((char *)&size_per_element, 8);
    reader.seekg(16, std::ios::cur);
    int32_t max_level;
    reader.read((char *)&max_level, 4);
    if (max_level > 1) {
      printf("Not supported\n");
      exit(-1);
    }
    eps.resize(1);
    reader.read((char *)&eps[0], 4);
    reader.seekg(8, std::ios::cur);
    size_t maxM0;
    reader.read((char *)&maxM0, 8);
    K = maxM0;
    reader.seekg(24, std::ios::cur);
    data = (node_t *)align_alloc((int64_t)N * K * 4, true, -1);
    for (int i = 0; i < N; ++i) {
      std::vector<char> buf(size_per_element);
      reader.read(buf.data(), size_per_element);
      int *lst = (int *)buf.data();
      int k = lst[0];
      memcpy(edges(i), lst + 1, k * 4);
    }
    
    init_ultra_compact();
    finalize_ultra_compact();
  }
};

} // namespace glass
