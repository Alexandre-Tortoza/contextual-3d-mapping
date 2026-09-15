"""Interpolação temporal de poses estimadas sem extrapolação implícita."""

from math import acos, sin, sqrt

from contextual_mapping_contracts import Pose, RigidTransform


# Amostra uma trajetória entre duas poses medidas. A translação é linear e a
# rotação segue o menor arco de quaternion; a aplicação preserva os dois
# timestamps de suporte e o fator usado na proveniência de projeção.
def interpolate_pose(before: Pose, after: Pose, timestamp_ns: int) -> Pose:
    """Interpola uma pose dentro de um intervalo de tempo estritamente crescente.

    Argumentos:
        before, after: poses no mesmo par de frames e relógio da aplicação.
        timestamp_ns: instante solicitado, em nanossegundos.
    Retorna:
        pose válida exatamente no instante solicitado.
    Levanta:
        ValueError: por frames diferentes, intervalo vazio ou extrapolação.
    """
    if type(timestamp_ns) is not int:
        raise TypeError("timestamp_ns must be an integer.")
    if not before.timestamp_ns < after.timestamp_ns:
        raise ValueError("Pose samples must have strictly increasing timestamps.")
    if not before.timestamp_ns <= timestamp_ns <= after.timestamp_ns:
        raise ValueError("Pose interpolation does not extrapolate.")
    left, right = before.transform, after.transform
    if (left.source_frame, left.target_frame) != (right.source_frame, right.target_frame):
        raise ValueError("Interpolated poses must use the same frames.")
    fraction = (timestamp_ns - before.timestamp_ns) / (after.timestamp_ns - before.timestamp_ns)
    translation = tuple(a + fraction * (b - a) for a, b in zip(left.translation_m, right.translation_m, strict=True))
    q0, q1 = left.rotation_xyzw, right.rotation_xyzw
    dot = sum(a * b for a, b in zip(q0, q1, strict=True))
    if dot < 0:
        q1, dot = tuple(-value for value in q1), -dot
    dot = min(1.0, max(0.0, dot))
    if dot > 1 - 1e-8:
        quaternion = tuple(a + fraction * (b - a) for a, b in zip(q0, q1, strict=True))
    else:
        angle = acos(dot)
        weights = sin((1 - fraction) * angle) / sin(angle), sin(fraction * angle) / sin(angle)
        quaternion = tuple(weights[0] * a + weights[1] * b for a, b in zip(q0, q1, strict=True))
    norm = sqrt(sum(value * value for value in quaternion))
    rotation = tuple(value / norm for value in quaternion)
    return Pose(RigidTransform(left.source_frame, left.target_frame, translation, rotation), timestamp_ns)
