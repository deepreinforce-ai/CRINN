#pragma once

#include "glass/searcher/graph_searcher.hpp"
#include "glass/searcher/refiner.hpp"
#include "glass/quant/quant.hpp"
#include <memory>

namespace glass {


namespace params {

  constexpr inline bool SQ8_REFINE = true;
  constexpr inline bool SQ8U_REFINE = true;
  constexpr inline bool SQ8P_REFINE = true;
  constexpr inline bool SQ4U_REFINE = true;
  constexpr inline bool SQ4UA_REFINE = true;
  constexpr inline bool SQ2U_REFINE = true;
  constexpr inline bool SQ1_REFINE = true;
  constexpr inline bool PQ8_REFINE = true;
  
  constexpr inline float SQ8_REFINE_FACTOR = 1.5f;
  constexpr inline float SQ8U_REFINE_FACTOR = 1.5f;
  constexpr inline float SQ8P_REFINE_FACTOR = 1.5f;
  constexpr inline float SQ4U_REFINE_FACTOR = 1.5f;
  constexpr inline float SQ4UA_REFINE_FACTOR = 1.5f;
  constexpr inline float SQ2U_REFINE_FACTOR = 3.0f;
  constexpr inline float SQ1_REFINE_FACTOR = 3.0f;
  constexpr inline float PQ8_REFINE_FACTOR = 1.5f;
  
  template <Metric metric> using RefineQuantizer = FP16Quantizer<metric>;
  
} // namespace params

inline std::unique_ptr<GraphSearcherBase>
create_searcher(Graph<int32_t> graph, const std::string &metric,
                const std::string &quantizer = "FP16") {
  using RType = std::unique_ptr<GraphSearcherBase>;
  auto m = metric_map[metric];
  auto qua = quantizer_map[quantizer];
  if (qua == QuantizerType::FP32) {
    if (m == Metric::L2) {
      return std::make_unique<GraphSearcher<FP32Quantizer<Metric::L2>>>(
          std::move(graph));
    } else if (m == Metric::IP) {
      return std::make_unique<GraphSearcher<FP32Quantizer<Metric::IP>>>(
          std::move(graph));
    } else {
      printf("Metric not suppported\n");
      return nullptr;
    }
  } else if (qua == QuantizerType::FP16) {
    if (m == Metric::L2) {
      return std::make_unique<GraphSearcher<FP16Quantizer<Metric::L2>>>(
          std::move(graph));
    } else if (m == Metric::IP) {
      return std::make_unique<GraphSearcher<FP16Quantizer<Metric::IP>>>(
          std::move(graph));
    } else {
      printf("Metric not suppported\n");
      return nullptr;
    }
  } else if (qua == QuantizerType::BF16) {
    if (m == Metric::L2) {
      return std::make_unique<GraphSearcher<BF16Quantizer<Metric::L2>>>(
          std::move(graph));
    } else if (m == Metric::IP) {
      return std::make_unique<GraphSearcher<BF16Quantizer<Metric::IP>>>(
          std::move(graph));
    } else {
      printf("Metric not suppported\n");
      return nullptr;
    }
  } else if (qua == QuantizerType::E5M2) {
    if (m == Metric::L2) {
      return std::make_unique<Refiner<params::RefineQuantizer<Metric::L2>>>(
          std::make_unique<GraphSearcher<E5M2Quantizer<Metric::L2>>>(
              std::move(graph)));
    } else if (m == Metric::IP) {
      return std::make_unique<Refiner<params::RefineQuantizer<Metric::IP>>>(
          std::make_unique<GraphSearcher<E5M2Quantizer<Metric::IP>>>(
              std::move(graph)));
    } else {
      printf("Metric not suppported\n");
      return nullptr;
    }
  } else if (qua == QuantizerType::SQ8U) {
    if (m == Metric::L2) {
      RType ret =
          std::make_unique<GraphSearcher<SQ8QuantizerUniform<Metric::L2>>>(
              std::move(graph));
      if (params::SQ8U_REFINE) {
        ret = std::make_unique<Refiner<params::RefineQuantizer<Metric::L2>>>(
            std::move(ret), params::SQ8U_REFINE_FACTOR);
      }
      return ret;
    } else if (m == Metric::IP) {
      RType ret =
          std::make_unique<GraphSearcher<SQ8QuantizerUniform<Metric::IP>>>(
              std::move(graph));
      if (params::SQ8U_REFINE) {
        ret = std::make_unique<Refiner<params::RefineQuantizer<Metric::IP>>>(
            std::move(ret), params::SQ8U_REFINE_FACTOR);
      }
      return ret;
    } else {
      printf("Metric not suppported\n");
      return nullptr;
    }
  } else if (qua == QuantizerType::SQ8) {
    if (m == Metric::L2) {
      RType ret = std::make_unique<GraphSearcher<SQ8Quantizer<Metric::L2>>>(
          std::move(graph));
      if (params::SQ8_REFINE) {
        ret = std::make_unique<Refiner<params::RefineQuantizer<Metric::L2>>>(
            std::move(ret), params::SQ8_REFINE_FACTOR);
      }
      return ret;
    } else if (m == Metric::IP) {
      RType ret = std::make_unique<GraphSearcher<SQ8Quantizer<Metric::IP>>>(
          std::move(graph));
      if (params::SQ8_REFINE) {
        ret = std::make_unique<Refiner<params::RefineQuantizer<Metric::IP>>>(
            std::move(ret), params::SQ8_REFINE_FACTOR);
      }
      return ret;
    } else {
      printf("Metric not suppported\n");
      return nullptr;
    }
  } else if (qua == QuantizerType::SQ6) {
    if (m == Metric::L2) {
      return std::make_unique<GraphSearcher<SQ6Quantizer<Metric::L2>>>(
          std::move(graph));
    } else if (m == Metric::IP) {
      return std::make_unique<GraphSearcher<SQ6Quantizer<Metric::IP>>>(
          std::move(graph));
    } else {
      printf("Metric not suppported\n");
      return nullptr;
    }
  } else if (qua == QuantizerType::SQ4U) {
    if (m == Metric::L2) {
      RType ret =
          std::make_unique<GraphSearcher<SQ4QuantizerUniform<Metric::L2>>>(
              std::move(graph));
      if (params::SQ4U_REFINE) {
        ret = std::make_unique<Refiner<params::RefineQuantizer<Metric::L2>>>(
            std::move(ret), params::SQ4U_REFINE_FACTOR);
      }
      return ret;
    } else if (m == Metric::IP) {
      RType ret =
          std::make_unique<GraphSearcher<SQ4QuantizerUniform<Metric::IP>>>(
              std::move(graph));
      if (params::SQ4U_REFINE) {
        ret = std::make_unique<Refiner<params::RefineQuantizer<Metric::IP>>>(
            std::move(ret), params::SQ4U_REFINE_FACTOR);
      }
      return ret;
    } else {
      printf("Metric not suppported\n");
      return nullptr;
    }
  } else if (qua == QuantizerType::SQ4UA) {
    if (m == Metric::L2) {
      RType ret =
          std::make_unique<GraphSearcher<SQ4QuantizerUniformAsym<Metric::L2>>>(
              std::move(graph));
      if (params::SQ4UA_REFINE) {
        ret = std::make_unique<Refiner<params::RefineQuantizer<Metric::L2>>>(
            std::move(ret), params::SQ4UA_REFINE_FACTOR);
      }
      return ret;
    } else if (m == Metric::IP) {
      RType ret =
          std::make_unique<GraphSearcher<SQ4QuantizerUniformAsym<Metric::IP>>>(
              std::move(graph));
      if (params::SQ4UA_REFINE) {
        ret = std::make_unique<Refiner<params::RefineQuantizer<Metric::IP>>>(
            std::move(ret), params::SQ4UA_REFINE_FACTOR);
      }
      return ret;
    } else {
      printf("Metric not suppported\n");
      return nullptr;
    }
  } else if (qua == QuantizerType::SQ4) {
    if (m == Metric::L2) {
      return std::make_unique<GraphSearcher<SQ4Quantizer<Metric::L2>>>(
          std::move(graph));
    } else if (m == Metric::IP) {
      return std::make_unique<GraphSearcher<SQ4Quantizer<Metric::IP>>>(
          std::move(graph));
    } else {
      printf("Metric not suppported\n");
      return nullptr;
    }
  } else if (qua == QuantizerType::SQ2U) {
    if (m == Metric::L2) {
      RType ret =
          std::make_unique<GraphSearcher<SQ2QuantizerUniform<Metric::L2>>>(
              std::move(graph));
      if (params::SQ2U_REFINE) {
        ret = std::make_unique<Refiner<params::RefineQuantizer<Metric::L2>>>(
            std::move(ret), params::SQ2U_REFINE_FACTOR);
      }
      return ret;
    } else if (m == Metric::IP) {
      RType ret =
          std::make_unique<GraphSearcher<SQ2QuantizerUniform<Metric::L2>>>(
              std::move(graph));
      if (params::SQ2U_REFINE) {
        ret = std::make_unique<Refiner<params::RefineQuantizer<Metric::IP>>>(
            std::move(ret), params::SQ2U_REFINE_FACTOR);
      }
      return ret;
    } else {
      printf("Metric not suppported\n");
      return nullptr;
    }
  } else if (qua == QuantizerType::SQ1) {
    if (m == Metric::IP) {
      RType ret = std::make_unique<GraphSearcher<SQ1Quantizer<Metric::IP>>>(
          std::move(graph));
      if (params::SQ1_REFINE) {
        ret = std::make_unique<Refiner<params::RefineQuantizer<Metric::IP>>>(
            std::move(ret), params::SQ1_REFINE_FACTOR);
      }
      return ret;
    } else {
      printf("Metric not suppported\n");
      return nullptr;
    }
  } else if (qua == QuantizerType::PQ8) {
    if (m == Metric::L2) {
      RType ret = std::make_unique<GraphSearcher<ProductQuant<Metric::L2>>>(
          std::move(graph));
      if (params::PQ8_REFINE) {
        ret = std::make_unique<Refiner<params::RefineQuantizer<Metric::L2>>>(
            std::move(ret), params::PQ8_REFINE_FACTOR);
      }
      return ret;
    } else if (m == Metric::IP) {
      RType ret = std::make_unique<GraphSearcher<ProductQuant<Metric::IP>>>(
          std::move(graph));
      if (params::PQ8_REFINE) {
        ret = std::make_unique<Refiner<params::RefineQuantizer<Metric::IP>>>(
            std::move(ret), params::PQ8_REFINE_FACTOR);
      }
      return ret;
    } else {
      printf("Metric not suppported\n");
      return nullptr;
    }
  } else {
    printf("Quantizer type not supported\n");
    return nullptr;
  }
}

} // namespace glass 