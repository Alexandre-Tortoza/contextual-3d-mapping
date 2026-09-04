"""Fronteira de entrada de dataset normalizada e sincronização determinística."""

from .dataset import CanonicalObservation, MultimodalDatasetAdapter, SyntheticDatasetAdapter
from .filesystem import DatasetFilesystem
from .synchronization import SynchronizationConfig, SynchronizedObservationGroup, synchronize

__all__ = ["CanonicalObservation", "DatasetFilesystem", "MultimodalDatasetAdapter", "SynchronizationConfig", "SynchronizedObservationGroup", "SyntheticDatasetAdapter", "synchronize"]
