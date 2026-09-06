"""Adapters de elevação aprendida de features e fallback explícito (#208)."""

from __future__ import annotations

import dataclasses
import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np

from visual_perception.application.lifecycle import ModelLifecycleManager
from visual_perception.config import FeatureExtractionConfig
from visual_perception.domain.errors import BackendExecutionError, BackendUnavailableError
from visual_perception.domain.feature_map import (
    FeatureGeneration,
    FeatureMap,
    FeatureRepresentation,
    SamplingRule,
)
from visual_perception.domain.image_payload import ImagePayload
from visual_perception.infrastructure.adapters._runtime import (
    raise_backend_execution_error,
    require_module,
    resolve_device,
)
from visual_perception.ports.feature_extraction import DenseFeatureExtractor


# Implementa o caminho FeatUp/JBU pré-treinado para DINOv2-small. O adapter
# contém toda dependência de torch hub/FeatUp e devolve somente o contract
# público FeatureMap, mantendo o runtime externo fora da application.
class FeatUpDenseFeatureExtractionAdapter:
    """Produz um mapa elevado pelo JBU aprendido oficial do FeatUp.

    O checkpoint fornecido pelo FeatUp é específico do DINOv2-small com
    384 canais. Ele não compartilha espaço vetorial com o DINOv2-base de
    768 canais e essa diferença permanece explícita na proveniência.
    """

    # Recebe o lifecycle compartilhado sem carregar pesos durante a composição.
    def __init__(self, lifecycle: ModelLifecycleManager | None = None) -> None:
        """Inicializa o adapter sem baixar nem carregar FeatUp."""
        self._lifecycle = lifecycle or ModelLifecycleManager()

    # Executa backbone e JBU em uma resolução aspect-preserving dimensionada
    # antes da inferência para que o mapa final caiba no teto configurado.
    def extract(self, image: ImagePayload, config: FeatureExtractionConfig) -> FeatureMap:
        """Extrai features DINOv2-small elevadas pelo FeatUp/JBU.

        Argumentos:
            image: pixels RGB originais e suas dimensões canônicas.
            config: repositório/checkpoint, resolução e teto de memória.
        Retorna:
            mapa elevado com stride, grade de origem e proveniência aprendida.
        Levanta:
            BackendUnavailableError: quando a instalação opcional não existe.
            BackendExecutionError: quando o runtime falha ou viola o contract.
        """
        torch = require_module("torch", config.backend)
        device = resolve_device(torch, config.device, config.backend)
        input_height, input_width = featup_input_size(
            image.height,
            image.width,
            config.input_resolution or 224,
            dimension=384,
            max_bytes=config.max_feature_map_mb * 1024 * 1024,
        )
        try:
            model = self._get_model(torch, config, device)
            tensor = torch.from_numpy(np.asarray(image.pixels)).permute(2, 0, 1)
            tensor = tensor.unsqueeze(0).to(device=device, dtype=torch.float32) / 255.0
            tensor = torch.nn.functional.interpolate(
                tensor,
                size=(input_height, input_width),
                mode="bilinear",
                align_corners=False,
            )
            mean = torch.tensor((0.485, 0.456, 0.406), device=device).view(1, 3, 1, 1)
            std = torch.tensor((0.229, 0.224, 0.225), device=device).view(1, 3, 1, 1)
            normalized = (tensor - mean) / std
            with torch.inference_mode():
                output = model(normalized)
            if output.ndim != 4 or int(output.shape[0]) != 1 or int(output.shape[1]) != 384:
                raise BackendExecutionError(
                    f"FeatUp retornou shape incompatível, esperado (1, 384, H, W), obtido {tuple(output.shape)}."
                )
            data = output[0].permute(1, 2, 0).detach().float().cpu().numpy()
            grid_height, grid_width = int(data.shape[0]), int(data.shape[1])
            return FeatureMap(
                data=np.asarray(data, dtype=np.float32),
                stride_x=image.width / grid_width,
                stride_y=image.height / grid_height,
                dimension=384,
                model_id="featup-dinov2-small-jbu",
                representation=FeatureRepresentation.ELEVATED_GRID,
                interpolation=(
                    SamplingRule.BILINEAR
                    if config.upsampling == "bilinear"
                    else SamplingRule.NEAREST
                ),
                checkpoint=config.checkpoint,
                preprocessing=(
                    f"imagenet_normalization:aspect_preserving:{input_width}x{input_height}"
                ),
                upsampling_method="featup_jbu_stack_cocostuff",
                generation=FeatureGeneration.LEARNED_UPSAMPLER,
                source_grid_width=input_width // 14,
                source_grid_height=input_height // 14,
                upsampler_checkpoint=config.upsampler_checkpoint,
                upsampler_checkpoint_digest=config.upsampler_checkpoint_digest,
                valid_support=np.ones((grid_height, grid_width), dtype=np.bool_),
            )
        except (BackendExecutionError, BackendUnavailableError):
            raise
        except Exception as error:
            raise_backend_execution_error(config.backend, "a elevação aprendida FeatUp/JBU", error)

    # Carrega DINOv2-small e a arquitetura JBU compatível com o checkpoint
    # oficial pelo lifecycle. A convolução adaptativa percorre a vizinhança
    # em acumulação para evitar tanto extensões CUDA quanto o tensor unfold.
    def _get_model(self, torch: Any, config: FeatureExtractionConfig, device: str) -> Any:
        """Retorna o FeatUp versionado, em modo de avaliação, no device pedido."""

        # Constrói o runtime pesado somente quando o lifecycle solicita a key.
        def factory() -> Any:
            """Carrega backbone e pesos JBU da revisão registrada na configuração."""
            transformers = require_module("transformers", config.backend)
            return _build_featup_runtime(
                torch,
                transformers,
                config.upsampler_checkpoint,
                config.upsampler_checkpoint_digest,
            ).to(device).eval()

        return self._lifecycle.get_or_load(
            f"feature_extraction:{config.upsampler_repository}:dinov2-jbu:{device}", factory
        )


# Reimplementa o runtime mínimo publicado pelo FeatUp: DINOv2-small,
# ChannelNorm e quatro blocos JBU. Existe localmente para que a instalação
# não dependa das extensões C++/CUDA sem wheel do repositório upstream; os
# pesos continuam sendo o checkpoint oficial e a revisão fonte é registrada.
def _build_featup_runtime(
    torch: Any,
    transformers: Any,
    checkpoint_url: str,
    expected_checkpoint_digest: str,
) -> Any:
    """Constrói o FeatUp/JBU compatível e carrega seus pesos oficiais."""
    functional = torch.nn.functional

    # Implementa uma etapa de joint bilateral upsampling com operações
    # públicas de torch, matematicamente equivalente à AdaptiveConv upstream.
    class _JBULearnedRange(torch.nn.Module):  # type: ignore[misc]
        """Uma etapa 2x do JBU aprendido guiado pela imagem RGB."""

        # Declara as mesmas camadas e nomes usados no checkpoint oficial.
        def __init__(self, feature_dimension: int, radius: int = 3) -> None:
            """Inicializa projeções de range, correção e kernel espacial."""
            super().__init__()
            self.radius = radius
            self.diameter = radius * 2 + 1
            self.key_dimension = 32
            self.range_temp = torch.nn.Parameter(torch.tensor(0.0))
            self.range_proj = torch.nn.Sequential(
                torch.nn.Conv2d(3, self.key_dimension, 1, 1),
                torch.nn.GELU(),
                torch.nn.Dropout2d(0.1),
                torch.nn.Conv2d(self.key_dimension, self.key_dimension, 1, 1),
            )
            kernel_channels = self.diameter**2
            self.fixup_proj = torch.nn.Sequential(
                torch.nn.Conv2d(3 + kernel_channels, kernel_channels, 1, 1),
                torch.nn.GELU(),
                torch.nn.Dropout2d(0.1),
                torch.nn.Conv2d(kernel_channels, kernel_channels, 1, 1),
            )
            self.sigma_spatial = torch.nn.Parameter(torch.tensor(1.0))

        # Calcula os pesos aprendidos de range a partir da imagem guia.
        def _range_kernel(self, guidance: Any) -> Any:
            """Retorna o kernel de range normalizado para cada pixel."""
            batch, _channels, height, width = guidance.shape
            projected = self.range_proj(guidance)
            padded = functional.pad(projected, [self.radius] * 4, mode="reflect")
            queries = functional.unfold(padded, self.diameter).reshape(
                batch,
                self.key_dimension,
                self.diameter**2,
                height,
                width,
            )
            temperature = self.range_temp.exp().clamp_min(1e-4).clamp_max(1e4)
            scores = torch.einsum("bcphw,bchw->bphw", queries, projected)
            return functional.softmax(temperature * scores, dim=1)

        # Calcula a prior gaussiana espacial compartilhada pelos pixels.
        def _spatial_kernel(self, device: Any) -> Any:
            """Retorna o kernel espacial gaussiano aprendido."""
            axis = torch.linspace(-1, 1, self.diameter, device=device)
            grid_y, grid_x = torch.meshgrid(axis, axis, indexing="ij")
            squared_distance = grid_x.square() + grid_y.square()
            return torch.exp(-squared_distance / (2 * self.sigma_spatial**2)).reshape(
                1, self.diameter**2, 1, 1
            )

        # Eleva a fonte para a geometria da guia e aplica a convolução
        # adaptativa como soma de vizinhanças ponderadas, sem materializar o
        # tensor unfold que excederia a VRAM em mapas de alta resolução.
        def forward(self, source: Any, guidance: Any) -> Any:
            """Eleva ``source`` 2x usando filtros bilaterais aprendidos."""
            batch, _channels, height, width = guidance.shape
            kernel = self._range_kernel(guidance) * self._spatial_kernel(source.device)
            kernel = kernel / kernel.sum(1, keepdim=True).clamp(1e-7)
            kernel = kernel + 0.1 * self.fixup_proj(torch.cat((kernel, guidance), dim=1))
            elevated = functional.interpolate(
                source, size=(height, width), mode="bicubic", align_corners=False
            )
            padded = functional.pad(elevated, [self.radius] * 4, mode="reflect")
            result = torch.zeros_like(elevated)
            kernel_index = 0
            for row_offset in range(self.diameter):
                for column_offset in range(self.diameter):
                    neighbourhood = padded[
                        :,
                        :,
                        row_offset : row_offset + height,
                        column_offset : column_offset + width,
                    ]
                    result.add_(neighbourhood * kernel[:, kernel_index].unsqueeze(1))
                    kernel_index += 1
            return result

    # Encadeia as quatro etapas 2x e a projeção residual final com os mesmos
    # nomes de parâmetros do checkpoint `upsampler.*` oficial.
    class _JBUStack(torch.nn.Module):  # type: ignore[misc]
        """Upsampler FeatUp 16x composto por quatro etapas JBU 2x."""

        # Constrói os quatro blocos e a correção residual final.
        def __init__(self, feature_dimension: int) -> None:
            """Inicializa a pilha JBU para a dimensão do DINOv2-small."""
            super().__init__()
            self.up1 = _JBULearnedRange(feature_dimension)
            self.up2 = _JBULearnedRange(feature_dimension)
            self.up3 = _JBULearnedRange(feature_dimension)
            self.up4 = _JBULearnedRange(feature_dimension)
            self.fixup_proj = torch.nn.Sequential(
                torch.nn.Dropout2d(0.2),
                torch.nn.Conv2d(feature_dimension, feature_dimension, kernel_size=1),
            )

        # Aplica uma etapa usando uma versão da guia com exatamente 2x a
        # resolução da fonte corrente.
        def _upsample(self, source: Any, guidance: Any, stage: Any) -> Any:
            """Executa uma etapa JBU 2x com guia RGB redimensionada."""
            target = (int(source.shape[2]) * 2, int(source.shape[3]) * 2)
            reduced_guidance = functional.adaptive_avg_pool2d(guidance, target)
            return stage(source, reduced_guidance)

        # Executa as quatro elevações e a projeção residual treinada.
        def forward(self, source: Any, guidance: Any) -> Any:
            """Produz features 16x mais densas que a grade fonte."""
            source = self._upsample(source, guidance, self.up1)
            source = self._upsample(source, guidance, self.up2)
            source = self._upsample(source, guidance, self.up3)
            source = self._upsample(source, guidance, self.up4)
            return self.fixup_proj(source) * 0.1 + source

    # Compõe o backbone HF equivalente e a pilha JBU em uma única interface
    # de inferência usada pelo adapter.
    class _FeatUpRuntime(torch.nn.Module):  # type: ignore[misc]
        """Runtime DINOv2-small + ChannelNorm + JBU oficial."""

        # Carrega o backbone público e recebe o upsampler já inicializado.
        def __init__(self, upsampler: Any) -> None:
            """Inicializa backbone, normalização de canal e upsampler."""
            super().__init__()
            self.model = transformers.AutoModel.from_pretrained("facebook/dinov2-small")
            self.norm = torch.nn.LayerNorm(384)
            self.upsampler = upsampler

        # Extrai tokens espaciais, normaliza canais e aplica a elevação JBU.
        def forward(self, image: Any) -> Any:
            """Retorna o mapa BCHW elevado para a imagem normalizada."""
            output = self.model(pixel_values=image).last_hidden_state[:, 1:]
            height, width = int(image.shape[2]) // 14, int(image.shape[3]) // 14
            features = output.reshape(-1, height, width, 384)
            features = self.norm(features).permute(0, 3, 1, 2)
            return self.upsampler(features, image)

    upsampler = _JBUStack(384)
    checkpoint = torch.hub.load_state_dict_from_url(
        checkpoint_url,
        map_location="cpu",
        progress=True,
    )["state_dict"]
    checkpoint_path = Path(torch.hub.get_dir()) / "checkpoints" / checkpoint_url.rsplit("/", 1)[-1]
    measured_digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    if measured_digest != expected_checkpoint_digest:
        raise BackendExecutionError(
            "O digest do checkpoint FeatUp/JBU não corresponde à configuração versionada."
        )
    weights = {
        key.removeprefix("upsampler."): value
        for key, value in checkpoint.items()
        if key.startswith("upsampler.")
    }
    upsampler.load_state_dict(weights, strict=True)
    return _FeatUpRuntime(upsampler)


# Decora um extrator primário com uma alternativa observável. Existe para
# que indisponibilidade/OOM do FeatUp não seja silenciosa nem aborte uma
# validação que declarou explicitamente uma política de fallback.
class FallbackDenseFeatureExtractionAdapter:
    """Executa um fallback configurado e registra o motivo no FeatureMap."""

    # Recebe implementações substituíveis do mesmo port para manter a regra
    # de fallback independente de qualquer biblioteca de modelo.
    def __init__(
        self,
        primary: DenseFeatureExtractor,
        fallback: DenseFeatureExtractor,
    ) -> None:
        """Inicializa o decorator com extratores primário e alternativo."""
        self._primary = primary
        self._fallback = fallback

    # Tenta o caminho aprendido e só captura erros operacionais tipados; bugs
    # de programação continuam visíveis e nunca acionam substituição.
    def extract(self, image: ImagePayload, config: FeatureExtractionConfig) -> FeatureMap:
        """Extrai pelo primário ou retorna o fallback com motivo auditável."""
        try:
            return self._primary.extract(image, config)
        except (BackendUnavailableError, BackendExecutionError) as error:
            fallback_config = dataclasses.replace(
                config,
                backend=config.fallback_backend or "dinov2",
                checkpoint=config.fallback_checkpoint or config.checkpoint,
                fallback_backend=None,
                fallback_checkpoint=None,
            )
            result = self._fallback.extract(image, fallback_config)
            return dataclasses.replace(
                result,
                fallback_reason=f"{type(error).__name__}: {error}",
            )


# Dimensiona a entrada FeatUp antes de materializar tensors: o JBU eleva a
# grade de patches por fator 16, então o custo final é previsível. Se o teto
# não comporta a resolução pedida, reduz a maior aresta explicitamente.
def featup_input_size(
    height: int,
    width: int,
    long_edge: int,
    *,
    dimension: int,
    max_bytes: int,
    patch_size: int = 14,
    elevation_factor: int = 16,
) -> tuple[int, int]:
    """Retorna uma entrada aspect-preserving cujo mapa elevado cabe no teto.

    Argumentos:
        height: altura original.
        width: largura original.
        long_edge: maior aresta inicialmente desejada.
        dimension: canais do espaço de feature.
        max_bytes: teto para o array float32 final.
        patch_size: patch do backbone.
        elevation_factor: fator espacial do JBU.
    Retorna:
        altura e largura divisíveis pelo patch.
    Levanta:
        ValueError: quando nem uma grade de um patch cabe no teto.
    """
    if min(height, width, long_edge, dimension, max_bytes, patch_size, elevation_factor) <= 0:
        raise ValueError("FeatUp dimensions and memory budget must be positive.")
    requested_scale = long_edge / max(height, width)
    target_height = max(patch_size, round(height * requested_scale / patch_size) * patch_size)
    target_width = max(patch_size, round(width * requested_scale / patch_size) * patch_size)
    output_bytes = (
        (target_height // patch_size * elevation_factor)
        * (target_width // patch_size * elevation_factor)
        * dimension
        * 4
    )
    if output_bytes <= max_bytes:
        return target_height, target_width

    reduction = math.sqrt(max_bytes / output_bytes)
    target_height = max(patch_size, math.floor(target_height * reduction / patch_size) * patch_size)
    target_width = max(patch_size, math.floor(target_width * reduction / patch_size) * patch_size)
    reduced_bytes = (
        (target_height // patch_size * elevation_factor)
        * (target_width // patch_size * elevation_factor)
        * dimension
        * 4
    )
    if reduced_bytes > max_bytes:
        raise ValueError("The FeatUp feature-map budget is too small for one source patch.")
    return target_height, target_width
