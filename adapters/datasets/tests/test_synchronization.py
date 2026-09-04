import pytest

from contextual_mapping_adapters import SynchronizationConfig, SyntheticDatasetAdapter, synchronize
from support import manifest, observation


# Constrói uma SynchronizationConfig padrão (âncora lidar, tipos
# lidar/rgb/imu/pose) para os testes deste arquivo variarem só a tolerância.
def config(tolerance: int = 5) -> SynchronizationConfig:
    return SynchronizationConfig("lidar", ("lidar", "rgb", "imu", "pose"), tolerance)


# Confere o caso combinado: observação exata, observação dentro da
# tolerância e um tipo esperado totalmente ausente no mesmo grupo.
def test_exact_near_tolerance_and_missing_observations() -> None:
    group = synchronize((observation("lidar", "lidar", 0, 100), observation("rgb", "camera", 0, 100), observation("imu", "imu", 0, 105)), config())[0]
    assert [item.kind for item in group.observations] == ["lidar", "rgb", "imu"]
    assert group.missing_kinds == ("pose",)
    assert group.observation("imu").reference.timestamp.nanoseconds == 105


# Confere que uma observação fora da tolerância não é casada e o tipo fica
# explicitamente marcado como ausente, em vez de ser casada incorretamente.
def test_out_of_tolerance_is_explicitly_missing() -> None:
    group = synchronize((observation("lidar", "lidar", 0, 100), observation("rgb", "camera", 0, 106)), config())[0]
    assert group.missing_kinds == ("rgb", "imu", "pose")


# Confere que a ordem de entrada não afeta o resultado (mesmo grupo com
# input normal ou invertido) e que uma observação já usada em um grupo não
# é reaproveitada em outro.
def test_out_of_order_replay_is_deterministic_and_matches_are_not_reused() -> None:
    items = (observation("lidar", "lidar", 0, 100), observation("rgb", "camera", 0, 102), observation("lidar", "lidar", 1, 104))
    policy = SynchronizationConfig("lidar", ("lidar", "rgb"), 5)
    first = synchronize(items, policy)
    assert first == synchronize(reversed(items), policy)
    assert first[0].observation("rgb") is not None
    assert first[1].missing_kinds == ("rgb",)


# Confere que uma tolerância negativa é rejeitada na construção da config,
# antes de qualquer sincronização ser tentada.
def test_invalid_tolerance_fails_before_execution() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        config(-1)


# Confere a integração ponta a ponta: a saída de SyntheticDatasetAdapter
# alimenta `synchronize` diretamente, sem exigir nenhum tipo específico de
# dataset entre as duas camadas.
def test_adapter_output_synchronizes_without_dataset_specific_types() -> None:
    adapter = SyntheticDatasetAdapter(manifest(), (observation("imu", "imu", 0, 99), observation("rgb", "camera", 0, 101), observation("lidar", "lidar", 0, 100)))
    groups = synchronize(adapter.observations("sequence"), config())
    assert len(groups) == 1
    assert groups[0].missing_kinds == ("pose",)
