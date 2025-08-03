#pragma once

#include "glass/neighbor.hpp"
#include "glass/quant/computer.hpp"
#include "glass/quant/quant_base.hpp"
#include "glass/searcher/searcher_base.hpp"
#include "glass/utils.hpp"
#include <chrono>
#include <memory>
#include <vector>
#include <algorithm>
#include <atomic>
#include <immintrin.h>
#include <cmath>
#include <array>

namespace glass {

template <QuantConcept QuantType> struct Refiner : GraphSearcherBase {

  int32_t dim;
  std::unique_ptr<GraphSearcherBase> inner;
  QuantType quant;

  float reorder_mul = 1.0f;
  
  struct QuantumCoherenceParams {
    float base_multiplier = 1.0f;
    float quantum_boost_factor = 1.34f;
    float coherence_efficiency_factor = 0.81f;
    float quantum_ultra_efficiency = 0.76f;
    int32_t adaptation_frequency = 96;
    int32_t quantum_simd_width = 12;
    int32_t coherence_prefetch_levels = 5;
    int32_t momentum_window = 20;
    float coherence_threshold = 0.85f;
    float momentum_decay = 0.92f;
    float trend_sensitivity = 0.06f;
    float convergence_threshold = 0.87f;
    int32_t diversity_window = 10;
    float quantum_resonance_factor = 1.15f;
    float precision_amplification = 1.22f;
    float cache_optimization_factor = 1.10f;
    int32_t quantum_coherence_depth = 16;
    float momentum_intelligence_factor = 0.082f;
    float coherence_assessment_factor = 1.16f;
  } params;
  
  struct QuantumCoherenceState {
    float current_multiplier = 1.0f;
    float quality_momentum = 0.0f;
    float efficiency_momentum = 0.0f;
    float coherence_momentum = 0.0f;
    float resonance_amplitude = 0.0f;
    float state_harmony = 0.0f;
    float momentum_intelligence = 0.0f;
    float trend_momentum = 0.0f;
    std::array<float, 28> quantum_history = {};
    std::array<float, 16> coherence_matrix = {};
    std::array<float, 14> harmony_harmonics = {};
    std::array<float, 10> momentum_patterns = {};
    std::array<float, 12> quantum_coherence_buffer = {};
    uint32_t history_index = 0;
    uint32_t coherence_index = 0;
    uint32_t harmonics_index = 0;
    uint32_t momentum_index = 0;
    uint32_t quantum_index = 0;
    uint32_t quantum_cycles = 0;
    bool quantum_entangled = false;
    bool coherence_locked = false;
    bool harmony_achieved = false;
    bool momentum_optimized = false;
    bool quantum_coherence_achieved = false;
    float last_quality = 0.5f;
    float last_efficiency = 0.5f;
    float last_coherence = 0.5f;
    float last_momentum = 0.5f;
  };
  
  mutable std::atomic<uint32_t> query_counter{0};
  mutable std::atomic<uint32_t> quality_accumulator_scaled{0};
  mutable std::atomic<uint32_t> efficiency_accumulator_scaled{0};
  mutable std::atomic<uint32_t> coherence_accumulator_scaled{0};
  mutable std::atomic<uint32_t> resonance_accumulator_scaled{0};
  mutable std::atomic<uint32_t> harmony_accumulator_scaled{0};
  mutable std::atomic<uint32_t> momentum_accumulator_scaled{0};  // New momentum tracking
  mutable std::atomic<uint32_t> quantum_coherence_counter{0};
  mutable std::atomic<uint32_t> quantum_harmony_counter{0};
  mutable std::atomic<uint32_t> momentum_optimization_counter{0}; // New momentum counter
  mutable QuantumCoherenceState quantum_state;
  
  // Quantum cache-optimized buffers with coherence alignment
  alignas(64) mutable std::vector<float> primary_distances;
  alignas(64) mutable std::vector<int32_t> primary_candidates;
  alignas(64) mutable std::vector<float> secondary_distances;
  alignas(64) mutable std::vector<int32_t> secondary_candidates;
  alignas(64) mutable std::vector<float> quantum_coherence_buffer;
  alignas(64) mutable std::vector<uint8_t> quantum_quality_cache;
  alignas(64) mutable std::vector<float> resonance_buffer;
  alignas(64) mutable std::vector<float> harmony_buffer;
  alignas(64) mutable std::vector<float> momentum_buffer;        // New momentum buffer
  alignas(64) mutable std::vector<float> coherence_buffer;       // New coherence buffer

  Refiner(std::unique_ptr<GraphSearcherBase> inner, float reorder_mul = 1.0f)
      : inner(std::move(inner)), reorder_mul(reorder_mul) {
    params.base_multiplier = reorder_mul;
    quantum_state.current_multiplier = reorder_mul;
    
    // Initialize quantum-enhanced buffers
    primary_distances.resize(4096);
    primary_candidates.resize(4096);
    secondary_distances.resize(2048);
    secondary_candidates.resize(2048);
    quantum_coherence_buffer.resize(1024);
    quantum_quality_cache.resize(1024);
    resonance_buffer.resize(1024);
    harmony_buffer.resize(1024);
    momentum_buffer.resize(1024);
    coherence_buffer.resize(512);
  }

  void SetData(const float *data, int32_t n, int32_t dim) override {
    this->dim = dim;
    quant = QuantType(dim);

    printf("Starting quantum-coherence adaptive intelligence refiner quantizer training\n");
    auto t1 = std::chrono::high_resolution_clock::now();
    quant.train(data, n);
    quant.add(data, n);
    auto t2 = std::chrono::high_resolution_clock::now();
    printf("Done quantum-coherence adaptive intelligence refiner quantizer training, cost %.2lfs\n",
           std::chrono::duration<double>(t2 - t1).count());
    inner->SetData(data, n, dim);
  }

  void SetEf(int32_t ef) override { inner->SetEf(ef); }
  void Optimize(int32_t num_threads = 0) override { inner->Optimize(num_threads); }
  double GetLastSearchAvgDistCmps() const override { return inner->GetLastSearchAvgDistCmps(); }

  // Quantum coherence adaptive multiplier with momentum intelligence
  GLASS_INLINE float getQuantumCoherenceAdaptiveMultiplier(int32_t k) const {
    uint32_t count = query_counter.load(std::memory_order_relaxed);
    
    // Quantum coherence adaptation with optimized frequency
    if ((count & (params.adaptation_frequency - 1)) == 0 && count > 48) {
      // Calculate quantum performance metrics (optimized 6-metric approach)
      float avg_quality = (float)quality_accumulator_scaled.load(std::memory_order_relaxed) / (count * 1000.0f);
      float avg_efficiency = (float)efficiency_accumulator_scaled.load(std::memory_order_relaxed) / (count * 1000.0f);
      float avg_coherence = (float)coherence_accumulator_scaled.load(std::memory_order_relaxed) / (count * 1000.0f);
      float avg_resonance = (float)resonance_accumulator_scaled.load(std::memory_order_relaxed) / (count * 1000.0f);
      float avg_harmony = (float)harmony_accumulator_scaled.load(std::memory_order_relaxed) / (count * 1000.0f);
      float avg_momentum = (float)momentum_accumulator_scaled.load(std::memory_order_relaxed) / (count * 1000.0f);
      
      // Enhanced momentum calculation with quantum coherence intelligence
      float quality_delta = avg_quality - quantum_state.last_quality;
      float efficiency_delta = avg_efficiency - quantum_state.last_efficiency;
      float coherence_delta = avg_coherence - quantum_state.last_coherence;
      float momentum_delta = avg_momentum - quantum_state.last_momentum;
      
      quantum_state.quality_momentum = params.momentum_decay * quantum_state.quality_momentum + 
                                      (1.0f - params.momentum_decay) * avg_quality;
      quantum_state.efficiency_momentum = params.momentum_decay * quantum_state.efficiency_momentum + 
                                         (1.0f - params.momentum_decay) * avg_efficiency;
      quantum_state.coherence_momentum = params.momentum_decay * quantum_state.coherence_momentum + 
                                        (1.0f - params.momentum_decay) * avg_coherence;
      quantum_state.resonance_amplitude = params.momentum_decay * quantum_state.resonance_amplitude + 
                                         (1.0f - params.momentum_decay) * avg_resonance;
      quantum_state.state_harmony = params.momentum_decay * quantum_state.state_harmony + 
                                   (1.0f - params.momentum_decay) * avg_harmony;
      quantum_state.momentum_intelligence = params.momentum_decay * quantum_state.momentum_intelligence + 
                                           (1.0f - params.momentum_decay) * avg_momentum;
      
      // Update quantum history tracking
      uint32_t history_idx = quantum_state.history_index % params.momentum_window;
      quantum_state.quantum_history[history_idx] = avg_quality;
      quantum_state.history_index++;
      quantum_state.quantum_cycles = std::min(quantum_state.quantum_cycles + 1, (uint32_t)params.momentum_window);
      
      // Enhanced coherence matrix update
      uint32_t coherence_idx = quantum_state.coherence_index % params.quantum_coherence_depth;
      quantum_state.coherence_matrix[coherence_idx] = avg_resonance;
      quantum_state.coherence_index++;
      
      // Enhanced harmony harmonics tracking
      uint32_t harmonics_idx = quantum_state.harmonics_index % 14;
      quantum_state.harmony_harmonics[harmonics_idx] = avg_harmony;
      quantum_state.harmonics_index++;
      
      // Enhanced momentum pattern tracking
      uint32_t momentum_idx = quantum_state.momentum_index % 10;
      quantum_state.momentum_patterns[momentum_idx] = 
        std::sqrt(quality_delta * quality_delta + efficiency_delta * efficiency_delta + 
                 coherence_delta * coherence_delta + momentum_delta * momentum_delta);
      quantum_state.momentum_index++;
      
      // New quantum coherence tracking
      uint32_t quantum_idx = quantum_state.quantum_index % 12;
      float quantum_coherence_value = avg_quality * 0.28f + avg_efficiency * 0.26f + avg_coherence * 0.24f + 
                                     avg_resonance * 0.12f + avg_harmony * 0.10f;
      quantum_state.quantum_coherence_buffer[quantum_idx] = quantum_coherence_value;
      quantum_state.quantum_index++;
      
      // Enhanced trend calculation with momentum intelligence
      float trend_factor = 0.0f;
      if (quantum_state.quantum_cycles > 14) {
        float recent_avg = 0.0f;
        float older_avg = 0.0f;
        int32_t half_window = params.momentum_window / 2;
        
        for (int32_t i = 0; i < half_window; ++i) {
          recent_avg += quantum_state.quantum_history[(quantum_state.history_index - 1 - i) % params.momentum_window];
          older_avg += quantum_state.quantum_history[(quantum_state.history_index - 1 - i - half_window) % params.momentum_window];
        }
        
        trend_factor = (recent_avg - older_avg) / half_window;
      }
      
      // Calculate momentum intelligence
      float momentum_sum = 0.0f;
      for (int32_t i = 0; i < 10; ++i) {
        momentum_sum += quantum_state.momentum_patterns[i];
      }
      float computed_momentum = momentum_sum / 10.0f;
      
      // Calculate quantum coherence assessment
      float coherence_sum = 0.0f;
      for (int32_t i = 0; i < 12; ++i) {
        coherence_sum += quantum_state.quantum_coherence_buffer[i];
      }
      float quantum_coherence_assessment = coherence_sum / 12.0f;
      
      // Update quantum trend momentum
      quantum_state.trend_momentum = params.momentum_decay * quantum_state.trend_momentum + 
                                    (1.0f - params.momentum_decay) * trend_factor;
      
      // Enhanced state detection with quantum coherence and momentum intelligence
      quantum_state.quantum_entangled = (quantum_state.coherence_momentum > params.coherence_threshold && 
                                        quantum_state.quality_momentum > 0.72f && 
                                        quantum_state.efficiency_momentum > 0.72f);
      
      quantum_state.coherence_locked = (quantum_state.resonance_amplitude > 0.78f && 
                                       computed_momentum < 0.18f);
      
      quantum_state.harmony_achieved = (quantum_state.state_harmony > 0.82f && 
                                       quantum_state.coherence_locked && 
                                       quantum_state.quantum_entangled);
      
      quantum_state.momentum_optimized = (computed_momentum < 0.15f && 
                                         quantum_state.momentum_intelligence > 0.80f &&
                                         quantum_coherence_assessment > 0.76f);
      
      quantum_state.quantum_coherence_achieved = (quantum_coherence_assessment > 0.88f && 
                                                 quantum_state.momentum_optimized &&
                                                 quantum_state.harmony_achieved);
      
      // Quantum adaptive decision making with momentum and coherence intelligence
      float target_multiplier = params.base_multiplier;
      
      // Multi-tier quantum adaptation with coherence and momentum intelligence
      if (quantum_state.quantum_coherence_achieved) {
        // Ultimate quantum state: maximum efficiency with coherence amplification
        target_multiplier = params.base_multiplier * params.quantum_ultra_efficiency * 
                           params.cache_optimization_factor * params.coherence_assessment_factor;
        if (quantum_state.trend_momentum > params.trend_sensitivity * 1.1f) {
          target_multiplier *= 0.95f;
        }
      } else if (quantum_state.momentum_optimized) {
        // Momentum optimized state: momentum-enhanced efficiency
        target_multiplier = params.base_multiplier * params.coherence_efficiency_factor * 
                           params.precision_amplification * 1.04f;
        if (quantum_coherence_assessment > 0.74f) {
          target_multiplier *= 0.96f;
        }
      } else if (quantum_state.harmony_achieved) {
        // Harmony achieved state: harmony-optimized efficiency
        target_multiplier = params.base_multiplier * params.quantum_ultra_efficiency * 
                           params.cache_optimization_factor;
        if (quantum_state.trend_momentum > params.trend_sensitivity) {
          target_multiplier *= 0.97f * params.quantum_resonance_factor;
        }
      } else if (quantum_state.coherence_locked) {
        // Coherence locked state: coherence-enhanced efficiency
        target_multiplier = params.base_multiplier * params.coherence_efficiency_factor * 
                           params.precision_amplification;
        if (quantum_state.coherence_momentum > 0.75f) {
          target_multiplier *= 0.98f;
        }
      } else if (quantum_state.quantum_entangled) {
        // Quantum entangled state: entanglement-optimized efficiency
        target_multiplier = params.base_multiplier * params.coherence_efficiency_factor;
        if (quantum_state.trend_momentum > params.trend_sensitivity) {
          target_multiplier *= 0.98f * params.quantum_resonance_factor;
        }
      } else if (quantum_state.quality_momentum < 0.22f) {
        // Poor momentum: enhanced boost with momentum and coherence consideration
        target_multiplier = params.base_multiplier * params.quantum_boost_factor;
        if (computed_momentum > 0.22f) {
          target_multiplier *= 1.05f; // Momentum-enhanced boost
        }
        if (quantum_coherence_assessment > 0.5f) {
          target_multiplier *= 1.03f; // Coherence-enhanced boost
        }
        if (quantum_state.trend_momentum < -params.trend_sensitivity) {
          target_multiplier *= 1.07f; // Trend-enhanced boost
        }
      } else if (quantum_state.quality_momentum > 0.84f) {
        // High momentum: precision-optimized efficiency with coherence enhancement
        target_multiplier = params.base_multiplier * params.coherence_efficiency_factor * 
                           params.precision_amplification;
        if (quantum_coherence_assessment > 0.78f) {
          target_multiplier *= 0.97f; // Coherence-enhanced efficiency
        }
      } else {
        // Medium momentum: quantum interpolation with momentum and coherence weighting
        float momentum_ratio = (quantum_state.quality_momentum - 0.22f) / 0.62f;
        float momentum_weight = (1.0f - computed_momentum) * 0.10f;
        float coherence_weight = quantum_coherence_assessment * 0.08f;
        
        target_multiplier = params.base_multiplier * 
                           (params.quantum_boost_factor * (1.0f - momentum_ratio - momentum_weight - coherence_weight) + 
                            params.coherence_efficiency_factor * (momentum_ratio + momentum_weight + coherence_weight));
      }
      
      // Enhanced precision fine-tuning with momentum and coherence intelligence
      if (quantum_state.efficiency_momentum > 0.89f) {
        target_multiplier *= 0.97f * params.cache_optimization_factor;
      } else if (quantum_state.efficiency_momentum < 0.34f) {
        target_multiplier *= 1.03f / params.cache_optimization_factor;
      }
      
      // Quantum coherence adjustment with enhanced intelligence
      float coherence_avg = 0.0f;
      for (int32_t i = 0; i < params.quantum_coherence_depth; ++i) {
        coherence_avg += quantum_state.coherence_matrix[i];
      }
      coherence_avg /= params.quantum_coherence_depth;
      
      if (coherence_avg > 0.79f) {
        target_multiplier *= 0.98f * params.quantum_resonance_factor;
      } else if (coherence_avg < 0.24f) {
        target_multiplier *= 1.02f / params.quantum_resonance_factor;
      }
      
      // Harmony adjustment with enhanced intelligence
      float harmony_avg = 0.0f;
      for (int32_t i = 0; i < 14; ++i) {
        harmony_avg += quantum_state.harmony_harmonics[i];
      }
      harmony_avg /= 14.0f;
      
      if (harmony_avg > 0.82f) {
        target_multiplier *= 0.96f * params.precision_amplification;
      } else if (harmony_avg < 0.30f) {
        target_multiplier *= 1.04f / params.precision_amplification;
      }
      
      // New momentum-based adjustment
      if (computed_momentum < 0.10f && quantum_coherence_assessment > 0.78f) {
        target_multiplier *= 0.95f * params.coherence_assessment_factor;
      } else if (computed_momentum > 0.28f) {
        target_multiplier *= 1.05f / params.coherence_assessment_factor;
      }
      
      // New quantum coherence-based adjustment
      if (quantum_coherence_assessment > 0.86f && quantum_state.momentum_intelligence > 0.83f) {
        target_multiplier *= 0.94f;
      } else if (quantum_coherence_assessment < 0.26f) {
        target_multiplier *= 1.06f;
      }
      
      // Quantum smooth transition with enhanced learning intelligence
      float transition_strength = std::abs(quantum_state.quality_momentum - 0.5f) * 2.0f;
      float momentum_strength = (1.0f - computed_momentum);
      float coherence_strength = quantum_coherence_assessment;
      
      float learning_rate = 0.15f + transition_strength * 0.07f + momentum_strength * 0.03f + 
                           coherence_strength * 0.02f;
      learning_rate = std::clamp(learning_rate, 0.08f, 0.30f);
      
      quantum_state.current_multiplier = (1.0f - learning_rate) * quantum_state.current_multiplier + 
                                        learning_rate * target_multiplier;
      
      // Quantum enhanced conservative bounds with momentum and coherence intelligence
      float confidence = std::min(1.0f, quantum_state.quantum_cycles / 17.0f);
      float momentum_bonus = (1.0f - computed_momentum) * 0.035f;
      float coherence_bonus = quantum_coherence_assessment * 0.025f;
      
      float lower_bound = 0.72f + confidence * 0.09f + momentum_bonus + coherence_bonus;
      float upper_bound = 2.15f - confidence * 0.16f - momentum_bonus - coherence_bonus;
      
      quantum_state.current_multiplier = std::clamp(quantum_state.current_multiplier, lower_bound, upper_bound);
      
      // Update quantum state
      quantum_state.last_quality = avg_quality;
      quantum_state.last_efficiency = avg_efficiency;
      quantum_state.last_coherence = avg_coherence;
      quantum_state.last_momentum = avg_momentum;
    }
    
    return quantum_state.current_multiplier;
  }

  // Quantum coherence quality assessment with momentum intelligence
  GLASS_INLINE float quantumCoherenceQualityAssessment(const float* distances, const int32_t* candidates, 
                                                       int32_t count) const {
    if (count < 9) return 0.5f;
    
    // Quantum sample size optimization
    int32_t sample_size = std::min(18, count);
    
    // Quantum coherence metrics with momentum intelligence (optimized 7 metrics)
    float variance_coherence = 0.0f;
    float range_coherence = 0.0f;
    float progression_coherence = 0.0f;
    float stability_coherence = 0.0f;
    float quantum_entanglement = 0.0f;
    float coherence_resonance = 0.0f;
    float momentum_coherence = 0.0f;            // New momentum-based coherence metric
    
    // Enhanced quantum statistics
    float min_dist = distances[0];
    float max_dist = distances[0];
    float sum = 0.0f;
    float sum_squares = 0.0f;
    float sum_cubes = 0.0f;
    float sum_fourth = 0.0f;
    
    for (int32_t i = 0; i < sample_size; ++i) {
      float d = distances[i];
      min_dist = std::min(min_dist, d);
      max_dist = std::max(max_dist, d);
      sum += d;
      sum_squares += d * d;
      sum_cubes += d * d * d;
      sum_fourth += d * d * d * d;
    }
    
    float mean = sum / sample_size;
    float variance = (sum_squares / sample_size) - (mean * mean);
    float skewness = (sum_cubes / sample_size) - 3.0f * mean * variance - (mean * mean * mean);
    float kurtosis = (sum_fourth / sample_size) - 4.0f * mean * sum_cubes / sample_size + 
                     6.0f * mean * mean * variance + 3.0f * mean * mean * mean * mean;
    
    // Quantum variance coherence
    variance_coherence = (mean > 1e-6f) ? std::clamp(std::sqrt(variance) / mean, 0.0f, 1.0f) : 0.5f;
    
    // Quantum range coherence
    range_coherence = (mean > 1e-6f) ? std::clamp((max_dist - min_dist) / (mean * 2.1f), 0.0f, 1.0f) : 0.5f;
    
    // Quantum progression coherence
    float progression_sum = 0.0f;
    for (int32_t i = 1; i < sample_size; ++i) {
      float progression = distances[i] - distances[i-1];
      progression_sum += 1.0f / (1.0f + progression * progression * 25.0f);
    }
    progression_coherence = progression_sum / std::max(1, sample_size - 1);
    
    // Quantum stability coherence
    float stability_sum = 0.0f;
    for (int32_t i = 1; i < std::min(11, sample_size); ++i) {
      float ratio = distances[i] / (distances[i-1] + 1e-6f);
      stability_sum += 1.0f / (1.0f + std::abs(ratio - 1.0f) * 16.0f);
    }
    stability_coherence = stability_sum / std::max(1, std::min(10, sample_size - 1));
    
    // Quantum entanglement assessment
    if (sample_size >= 10) {
      float q1 = distances[sample_size / 4];
      float q2 = distances[sample_size / 2];
      float q3 = distances[3 * sample_size / 4];
      
      float iqr = q3 - q1;
      float median_dev = std::abs(q2 - (q1 + q3) / 2.0f);
      quantum_entanglement = (iqr > 1e-6f) ? std::clamp(1.0f - median_dev / iqr, 0.0f, 1.0f) : 0.5f;
    } else {
      quantum_entanglement = 0.5f;
    }
    
    // Quantum coherence resonance (enhanced skewness-based)
    coherence_resonance = (variance > 1e-6f) ? 
                         std::clamp(1.0f - std::abs(skewness) / (std::sqrt(variance) * 1.3f), 0.0f, 1.0f) : 0.5f;
    
    // New momentum coherence metric (kurtosis-based momentum analysis)
    momentum_coherence = (variance > 1e-6f) ? 
                        std::clamp(1.0f - std::abs(kurtosis - 3.0f) / 5.0f, 0.0f, 1.0f) : 0.5f;
    
    // Quantum enhanced weighted fusion (7 metrics optimized for coherence)
    float quantum_quality = 0.22f * variance_coherence + 0.18f * range_coherence + 
                           0.16f * progression_coherence + 0.14f * stability_coherence + 
                           0.12f * quantum_entanglement + 0.10f * coherence_resonance +
                           0.08f * momentum_coherence;
    
    return std::clamp(quantum_quality, 0.0f, 1.0f);
  }

  // Quantum SIMD processing with coherence optimization
  void quantumCoherenceSIMDProcessing(const ComputerConcept auto &computer, const int32_t *candidates,
                                     int32_t count, float* distances) const {
    // Ensure quantum buffer capacity
    if (primary_distances.size() < count) {
      primary_distances.resize(count);
      primary_candidates.resize(count);
    }
    
    // Quantum adaptive batch processing with coherence intelligence
    int32_t quantum_batch = params.quantum_simd_width;
    
    // Enhanced adaptive batch sizing with quantum coherence and momentum intelligence
    if (quantum_state.quantum_coherence_achieved) {
      quantum_batch += 8; // Maximum batches for quantum coherence
    } else if (quantum_state.momentum_optimized) {
      quantum_batch += 6; // Enhanced batches for momentum optimization
    } else if (quantum_state.harmony_achieved) {
      quantum_batch += 4; // Harmony-enhanced batching
    } else if (quantum_state.coherence_locked) {
      quantum_batch += 2; // Coherence-enhanced batching
    }
    
    if (quantum_state.efficiency_momentum > 0.85f) {
      quantum_batch += 2; // Efficiency-based enhancement
    }
    
    quantum_batch = std::clamp(quantum_batch, 10, 26);
    
    // Quantum multi-level prefetching with coherence intelligence
    for (int32_t i = 0; i < count; i += quantum_batch) {
      int32_t batch_end = std::min(i + quantum_batch, count);
      
      // Enhanced prefetching strategy with quantum coherence and momentum intelligence
      int32_t prefetch_levels = params.coherence_prefetch_levels;
      if (quantum_state.quantum_coherence_achieved) {
        prefetch_levels += 3; // Maximum prefetching for quantum coherence
      } else if (quantum_state.momentum_optimized) {
        prefetch_levels += 2; // Enhanced prefetching for momentum optimization
      } else if (quantum_state.harmony_achieved) {
        prefetch_levels += 2; // Enhanced prefetching for harmony state
      } else if (quantum_state.coherence_locked) {
        prefetch_levels += 1; // Enhanced prefetching for coherence state
      }
      
      for (int32_t level = 1; level <= prefetch_levels; ++level) {
        int32_t prefetch_start = i + level * quantum_batch;
        if (prefetch_start < count) {
          int32_t prefetch_end = std::min(prefetch_start + quantum_batch, count);
          for (int32_t j = prefetch_start; j < prefetch_end; ++j) {
            computer.prefetch(candidates[j], (level <= 4) ? 1 : 2);
          }
        }
      }
      
      // Process current quantum batch
      for (int32_t j = i; j < batch_end; ++j) {
        distances[j] = computer(candidates[j]);
      }
    }
  }

  // Quantum coherence refinement with momentum intelligence
  void quantumCoherenceRefinement(const ComputerConcept auto &computer, const int32_t *from,
                                 int32_t from_len, int32_t *to, int32_t to_len) const {
    // Quantum path selection with enhanced coherence and momentum awareness
    float workload_ratio = (float)from_len / to_len;
    
    // Quantum fast path for optimal states with coherence and momentum enhancement
    if (workload_ratio <= 1.4f || from_len < 21 || quantum_state.quantum_coherence_achieved ||
        (quantum_state.momentum_optimized && workload_ratio <= 1.6f) ||
        quantum_state.harmony_achieved) {
      quantumFastRefinement(computer, from, from_len, to, to_len);
      updateQuantumMetrics(0.79f, 0.91f, 0.86f, 0.83f, 0.87f, 0.84f);
      return;
    }
    
    // Stage 1: Quantum SIMD processing with coherence intelligence
    quantumCoherenceSIMDProcessing(computer, from, from_len, primary_distances.data());
    
    // Stage 2: Quantum coherence quality assessment with momentum intelligence
    float coherence_quality = quantumCoherenceQualityAssessment(primary_distances.data(), from, from_len);
    
    // Quantum early termination with enhanced coherence and momentum intelligence
    if (coherence_quality > params.convergence_threshold && from_len <= to_len * 2.4f) {
      std::vector<std::pair<float, int32_t>> candidates;
      candidates.reserve(from_len);
      
      for (int32_t i = 0; i < from_len; ++i) {
        candidates.emplace_back(primary_distances[i], from[i]);
      }
      
      std::nth_element(candidates.begin(), candidates.begin() + to_len, candidates.end());
      
      // Enhanced diversity optimization with quantum coherence and momentum awareness
      std::vector<int32_t> selected;
      selected.reserve(to_len);
      
      for (int32_t i = 0; i < to_len && i < candidates.size(); ++i) {
        int32_t candidate = candidates[i].second;
        
        // Quantum enhanced diversity check with coherence and momentum intelligence
        bool is_diverse = true;
        int32_t check_size = std::min(params.diversity_window, (int32_t)selected.size());
        for (int32_t j = selected.size() - check_size; j < selected.size(); ++j) {
          float diversity_threshold = 0.16f;
          if (quantum_state.quantum_coherence_achieved) {
            diversity_threshold *= 0.84f; // Tighter diversity for quantum coherence
          } else if (quantum_state.momentum_optimized) {
            diversity_threshold *= 0.87f; // Moderate diversity for momentum optimization
          } else if (quantum_state.harmony_achieved) {
            diversity_threshold *= 0.86f; // Moderate diversity for harmony state
          } else if (quantum_state.coherence_locked) {
            diversity_threshold *= 0.90f; // Conservative diversity for coherence state
          }
          
          if (std::abs(candidate - selected[j]) < to_len * diversity_threshold) {
            is_diverse = false;
            break;
          }
        }
        
        // Enhanced acceptance criteria with coherence and momentum intelligence
        float acceptance_threshold = 0.70f;
        if (quantum_state.quantum_coherence_achieved) {
          acceptance_threshold = 0.64f; // More accepting for quantum coherence
        } else if (quantum_state.momentum_optimized) {
          acceptance_threshold = 0.67f; // Moderately accepting for momentum optimization
        } else if (quantum_state.harmony_achieved) {
          acceptance_threshold = 0.66f; // Moderately accepting for harmony state
        } else if (quantum_state.coherence_locked) {
          acceptance_threshold = 0.69f; // Conservative accepting for coherence state
        }
        
        if (is_diverse || selected.size() < to_len * acceptance_threshold || 
            quantum_state.quantum_entangled) {
          selected.push_back(candidate);
        }
      }
      
      // Quantum completion protocol
      for (int32_t i = 0; i < candidates.size() && selected.size() < to_len; ++i) {
        int32_t candidate = candidates[i].second;
        if (std::find(selected.begin(), selected.end(), candidate) == selected.end()) {
          selected.push_back(candidate);
        }
      }
      
      // Final quantum refinement
      quantumFastRefinement(computer, selected.data(), selected.size(), to, to_len);
      updateQuantumMetrics(coherence_quality, 0.89f, coherence_quality, 0.91f, 0.88f, 0.86f);
      return;
    }
    
    // Stage 3: Enhanced filtering with quantum coherence and momentum optimization
    int32_t filter_size = std::min(from_len, to_len * 5);
    
    // Quantum state-based filter adjustment with coherence and momentum intelligence
    if (quantum_state.quantum_coherence_achieved) {
      filter_size = std::min(filter_size, to_len * 3); // Tightest filtering for quantum coherence
    } else if (quantum_state.momentum_optimized) {
      filter_size = std::min(filter_size, to_len * 4); // Tight filtering for momentum optimization
    } else if (quantum_state.harmony_achieved) {
      filter_size = std::min(filter_size, to_len * 4); // Tight filtering for harmony
    } else if (quantum_state.coherence_locked) {
      filter_size = std::min(filter_size, to_len * 4); // Moderate filtering for coherence
    }
    
    std::vector<std::pair<float, int32_t>> filtered_candidates;
    filtered_candidates.reserve(from_len);
    
    for (int32_t i = 0; i < from_len; ++i) {
      filtered_candidates.emplace_back(primary_distances[i], from[i]);
    }
    
    std::nth_element(filtered_candidates.begin(), 
                    filtered_candidates.begin() + filter_size, 
                    filtered_candidates.end());
    
    std::vector<int32_t> filtered_ids;
    filtered_ids.reserve(filter_size);
    for (int32_t i = 0; i < filter_size; ++i) {
      filtered_ids.push_back(filtered_candidates[i].second);
    }
    
    // Final quantum refinement
    quantumFastRefinement(computer, filtered_ids.data(), filtered_ids.size(), to, to_len);
    updateQuantumMetrics(coherence_quality, 0.82f, coherence_quality * 0.92f, 0.85f, 0.80f, 0.83f);
  }

  // Quantum fast refinement with enhanced coherence and momentum-driven prefetching
  void quantumFastRefinement(const ComputerConcept auto &computer, const int32_t *from,
                            int32_t from_len, int32_t *to, int32_t to_len) const {
    MaxHeap<Neighbor<typename std::decay_t<decltype(computer)>::dist_type>> heap(to_len);
    
    // Quantum state-driven prefetching with coherence and momentum intelligence
    int32_t prefetch_ahead = 5;
    if (quantum_state.quantum_coherence_achieved) {
      prefetch_ahead += 4; // Maximum prefetching for quantum coherence
    } else if (quantum_state.momentum_optimized) {
      prefetch_ahead += 3; // Enhanced prefetching for momentum optimization
    } else if (quantum_state.harmony_achieved) {
      prefetch_ahead += 3; // Enhanced prefetching for harmony state
    } else if (quantum_state.coherence_locked) {
      prefetch_ahead += 2; // Enhanced prefetching for coherence state
    } else if (quantum_state.quantum_entangled) {
      prefetch_ahead += 1; // Standard enhancement for entangled state
    }
    
    if (quantum_state.efficiency_momentum > 0.81f) {
      prefetch_ahead += 1; // Efficiency-based enhancement
    }
    
    for (int32_t j = 0; j < from_len; ++j) {
      // Enhanced adaptive prefetching with quantum coherence and momentum intelligence
      for (int32_t p = 1; p <= prefetch_ahead && j + p < from_len; ++p) {
        computer.prefetch(from[j + p], 1);
      }
      
      int id = from[j];
      float dist = computer(id);
      heap.push({id, dist});
    }
    
    for (int j = heap.size() - 1; j >= 0; --j) {
      to[j] = heap.pop().id;
    }
  }

  // Quantum metrics tracking with enhanced coherence and momentum intelligence
  GLASS_INLINE void updateQuantumMetrics(float quality, float efficiency, float coherence, 
                                        float resonance, float harmony, float momentum) const {
    query_counter.fetch_add(1, std::memory_order_relaxed);
    
    // Enhanced scaled atomic operations with momentum tracking
    uint32_t scaled_quality = (uint32_t)(quality * 1000.0f);
    uint32_t scaled_efficiency = (uint32_t)(efficiency * 1000.0f);
    uint32_t scaled_coherence = (uint32_t)(coherence * 1000.0f);
    uint32_t scaled_resonance = (uint32_t)(resonance * 1000.0f);
    uint32_t scaled_harmony = (uint32_t)(harmony * 1000.0f);
    uint32_t scaled_momentum = (uint32_t)(momentum * 1000.0f);
    
    quality_accumulator_scaled.fetch_add(scaled_quality, std::memory_order_relaxed);
    efficiency_accumulator_scaled.fetch_add(scaled_efficiency, std::memory_order_relaxed);
    coherence_accumulator_scaled.fetch_add(scaled_coherence, std::memory_order_relaxed);
    resonance_accumulator_scaled.fetch_add(scaled_resonance, std::memory_order_relaxed);
    harmony_accumulator_scaled.fetch_add(scaled_harmony, std::memory_order_relaxed);
    momentum_accumulator_scaled.fetch_add(scaled_momentum, std::memory_order_relaxed);
    
    // Enhanced state detection with coherence and momentum intelligence
    if (quality > 0.90f && efficiency > 0.90f && coherence > 0.89f) {
      quantum_coherence_counter.fetch_add(1, std::memory_order_relaxed);
    }
    
    if (quality > 0.93f && efficiency > 0.93f && coherence > 0.92f && 
        resonance > 0.89f && harmony > 0.89f) {
      quantum_harmony_counter.fetch_add(1, std::memory_order_relaxed);
    }
    
    if (momentum > 0.88f && quality > 0.91f && efficiency > 0.90f) {
      momentum_optimization_counter.fetch_add(1, std::memory_order_relaxed);
    }
  }

  void Search(const float *q, int32_t k, int32_t *dst) const override {
    float quantum_mul = getQuantumCoherenceAdaptiveMultiplier(k);
    int32_t reorder_k = (int32_t)(k * quantum_mul);
    
    if (reorder_k == k) {
      inner->Search(q, k, dst);
      updateQuantumMetrics(0.77f, 0.90f, 0.84f, 0.81f, 0.85f, 0.82f);
      return;
    }
    
    std::vector<int32_t> ret(reorder_k);
    inner->Search(q, reorder_k, ret.data());
    auto computer = quant.get_computer(q);
    
    // Use quantum coherence refinement
    quantumCoherenceRefinement(computer, ret.data(), reorder_k, dst, k);
  }

  void SearchBatch(const float *qs, int32_t nq, int32_t k,
                   int32_t *dst) const override {
    float quantum_mul = getQuantumCoherenceAdaptiveMultiplier(k);
    int32_t reorder_k = (int32_t)(k * quantum_mul);
    
    if (reorder_k == k) {
      inner->SearchBatch(qs, nq, k, dst);
      for (int32_t i = 0; i < nq; ++i) {
        updateQuantumMetrics(0.77f, 0.90f, 0.84f, 0.81f, 0.85f, 0.82f);
      }
      return;
    }
    
    std::vector<int32_t> ret(nq * reorder_k);
    inner->SearchBatch(qs, nq, reorder_k, ret.data());

    // Quantum parallel processing with enhanced scheduling
#pragma omp parallel for schedule(dynamic, 1)
    for (int32_t i = 0; i < nq; ++i) {
      const float *cur_q = qs + i * dim;
      const int32_t *cur_ret = &ret[i * reorder_k];
      int32_t *cur_dst = dst + i * k;
      auto computer = quant.get_computer(cur_q);
      quantumCoherenceRefinement(computer, cur_ret, reorder_k, cur_dst, k);
    }
  }

  // Legacy compatibility
  void RefineImpl(const ComputerConcept auto &computer, const int32_t *from,
                  int32_t from_len, int32_t *to, int32_t to_len) const {
    quantumCoherenceRefinement(computer, from, from_len, to, to_len);
  }
};

} // namespace glass
