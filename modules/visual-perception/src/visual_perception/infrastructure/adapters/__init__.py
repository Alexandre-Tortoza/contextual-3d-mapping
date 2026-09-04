"""Adapters de backend real selecionados pelo benchmark do módulo (#174).

Cada adapter implementa um port canônico e usa imports lazy para que o módulo
continue utilizável com os fakes quando os extras de ML, checkpoints ou GPU
não estiverem disponíveis.
"""

from visual_perception.infrastructure.adapters.factory import create_perception_ports
from visual_perception.infrastructure.adapters.feature_extraction_backend import (
    RealDenseFeatureExtractionAdapter,
)
from visual_perception.infrastructure.adapters.language_embedding_backend import (
    RealLanguageAlignedEncoderAdapter,
)
from visual_perception.infrastructure.adapters.multimodal_reasoning_backend import (
    RealMultimodalReasoningAdapter,
)
from visual_perception.infrastructure.adapters.region_discovery_backend import RealRegionDiscoveryAdapter

__all__ = [
    "RealDenseFeatureExtractionAdapter",
    "RealLanguageAlignedEncoderAdapter",
    "RealMultimodalReasoningAdapter",
    "RealRegionDiscoveryAdapter",
    "create_perception_ports",
]
