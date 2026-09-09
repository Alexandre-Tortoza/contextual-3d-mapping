"""Capacidade de mapeamento geométrico persistente."""
"""Fronteira pública do módulo geometric-map."""

from .in_memory import InMemoryGeometricMap
from .models import Bounds3D, GeometryPoint, GeometryReference

__all__ = ["Bounds3D", "GeometryPoint", "GeometryReference", "InMemoryGeometricMap"]
