"""Grounding por localização textual e SAM promptado, com modelos sequenciais."""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Any

import numpy as np

from visual_perception.application.lifecycle import ModelLifecycleManager
from visual_perception.application.semantic_grounding import box_mask
from visual_perception.application.support import fingerprint_of
from visual_perception.config import SemanticGroundingConfig
from visual_perception.domain.geometry import BoundingBox, Mask
from visual_perception.domain.grounding import GroundingPrediction, GroundingRequest, GroundingStatus
from visual_perception.domain.image_payload import ImagePayload
from visual_perception.domain.references import ModelProvenance
from visual_perception.domain.semantics import RegionKind
from visual_perception.infrastructure.adapters._runtime import payload_to_pil, require_module, resolve_device


# Contém as duas bibliotecas de modelo atrás de um único port de capacidade.
# Detector e SAM nunca ficam residentes simultaneamente; o embedding da imagem
# só permanece durante a etapa de segmentação daquele frame.
class RealSemanticGroundingAdapter:
    """Localiza o conceito, segmenta suas boxes e devolve geometria auditável."""

    # Compartilha o lifecycle dos backends existentes sem carregar pesos.
    def __init__(self, lifecycle: ModelLifecycleManager | None = None) -> None:
        """Recebe o dono da residência sequencial de modelos."""
        self._lifecycle = lifecycle or ModelLifecycleManager()

    # Localiza todos os conceitos antes de carregar SAM. Esse agrupamento
    # evita uma troca de modelos a cada região em máquinas com 8 GB de VRAM.
    def ground(
        self, image: ImagePayload, requests: tuple[GroundingRequest, ...], config: SemanticGroundingConfig,
    ) -> tuple[GroundingPrediction, ...]:
        """Produz uma predição ou falha por request, sem fallback de discovery."""
        try:
            return self._ground_frame(image, requests, config)
        finally:
            self._lifecycle.release_active()

    # Mantém referências locais de modelos fora da pilha que os libera,
    # inclusive se carregamento ou inferência falharem no meio do frame.
    def _ground_frame(
        self, image: ImagePayload, requests: tuple[GroundingRequest, ...], config: SemanticGroundingConfig,
    ) -> tuple[GroundingPrediction, ...]:
        """Localiza e segmenta requests; ground garante a liberação final."""
        if not requests:
            return ()
        provenance = (
            ModelProvenance("semantic_localization", "grounding_dino", fingerprint_of(config), checkpoint=config.detector_checkpoint),
            ModelProvenance("prompted_segmentation", "sam", fingerprint_of(config), checkpoint=config.segmenter_checkpoint),
        )
        detections, costs = self._detect(image, tuple(dict.fromkeys(item.concept for item in requests)), config)
        results: dict[str, GroundingPrediction] = {}
        selected: list[tuple[GroundingRequest, tuple[tuple[BoundingBox, float], ...]]] = []
        charged: set[str] = set()
        for request in requests:
            boxes, error = detections[request.concept]
            matching = tuple(
                (box, score) for box, score in boxes
                if (request.discovery_mask.data & box_mask(box, request.discovery_mask.data.shape)).any()
            )
            count = 0 if request.concept in charged else 1
            latency = 0.0 if request.concept in charged else costs[request.concept]
            charged.add(request.concept)
            reason = error or ("Nenhuma box localizada sobrepõe discovery." if not matching else None)
            if len(matching) > 1 and request.region_kind is not RegionKind.STUFF:
                reason = "Múltiplas localizações distintas; identidade espacial ambígua."
            results[request.region_id] = GroundingPrediction(
                request.region_id, request.concept, provenance=provenance,
                reason=reason, model_calls=count, latency_s=latency,
            )
            if reason is None:
                selected.append((request, matching))
        # _detect retorna somente objetos CPU. Sua pilha e referências ao
        # modelo terminaram antes da liberação, inclusive em caso de falha.
        self._lifecycle.release_active()
        if selected:
            try:
                self._segment(image, selected, results, config)
            finally:
                self._lifecycle.release_active()
        return tuple(results[item.region_id] for item in requests)

    # Um único conceito por consulta impede que labels concatenados sejam
    # reatribuídos por similaridade textual; requests repetidos compartilham a
    # mesma detecção. NMS só remove boxes quase idênticas do mesmo conceito.
    def _detect(
        self, image: ImagePayload, concepts: tuple[str, ...], config: SemanticGroundingConfig,
    ) -> tuple[dict[str, tuple[list[tuple[BoundingBox, float]], str | None]], dict[str, float]]:
        """Retorna boxes em pixels globais e custo medido por conceito."""
        torch = require_module("torch", "grounded_sam")
        transformers = require_module("transformers", "grounded_sam")
        device = resolve_device(torch, config.device, "grounded_sam")

        # Mantém modelo e processor juntos no slot gerenciado pelo lifecycle.
        def factory() -> Any:
            """Carrega somente o detector e seu processor do checkpoint declarado."""
            return (
                transformers.AutoModelForZeroShotObjectDetection.from_pretrained(config.detector_checkpoint).to(device).eval(),
                transformers.AutoProcessor.from_pretrained(config.detector_checkpoint),
            )

        load_started = time.perf_counter()
        model, processor = self._lifecycle.get_or_load(f"semantic_localization:{config.detector_checkpoint}:{device}", factory)
        load_s = time.perf_counter() - load_started
        pil = payload_to_pil(image, "grounded_sam")
        detections: dict[str, tuple[list[tuple[BoundingBox, float]], str | None]] = {}
        costs = {}
        for index, concept in enumerate(concepts):
            started = time.perf_counter()
            try:
                inputs = processor(images=pil, text=concept.strip().rstrip(".") + ".", return_tensors="pt").to(device)
                with torch.inference_mode():
                    outputs = model(**inputs)
                result = processor.post_process_grounded_object_detection(
                    outputs, inputs.input_ids, threshold=config.detection_threshold,
                    text_threshold=config.text_threshold, target_sizes=[(image.height, image.width)],
                )[0]
                boxes: list[tuple[BoundingBox, float]] = []
                for coordinates, score in sorted(zip(result["boxes"].cpu().tolist(), result["scores"].cpu().tolist(), strict=True), key=lambda item: -item[1]):
                    if not np.isfinite(coordinates).all():
                        continue
                    try:
                        box = BoundingBox(*coordinates).clipped_to(width=image.width, height=image.height)
                    except ValueError:
                        continue
                    if all(box.iou(previous) < config.duplicate_box_iou for previous, _ in boxes):
                        boxes.append((box, float(score)))
                detections[concept] = (boxes, None)
                del inputs, outputs, result
            except (RuntimeError, ValueError, TypeError) as error:
                detections[concept] = ([], f"Falha de localização: {error}")
            costs[concept] = time.perf_counter() - started + (load_s if index == 0 else 0)
        return detections, costs

    # Reutiliza um embedding SAM por frame, passando apenas boxes produzidas
    # pelo detector. O VLM nunca informa pontos, polígonos ou masks.
    def _segment(
        self, image: ImagePayload,
        selected: list[tuple[GroundingRequest, tuple[tuple[BoundingBox, float], ...]]],
        results: dict[str, GroundingPrediction], config: SemanticGroundingConfig,
    ) -> None:
        """Decodifica masks condicionadas às boxes e registra scores e latência."""
        torch = require_module("torch", "grounded_sam")
        transformers = require_module("transformers", "grounded_sam")
        device = resolve_device(torch, config.device, "grounded_sam")

        # Um segundo slot substitui o detector já liberado; não coexistem pesos.
        def factory() -> Any:
            """Carrega o mesmo SAM disponível para discovery, agora com prompts."""
            return (
                transformers.SamModel.from_pretrained(config.segmenter_checkpoint).to(device).eval(),
                transformers.SamProcessor.from_pretrained(config.segmenter_checkpoint),
            )

        started = time.perf_counter()
        model, processor = self._lifecycle.get_or_load(f"semantic_segmentation:{config.segmenter_checkpoint}:{device}", factory)
        pil = payload_to_pil(image, "grounded_sam")
        inputs = processor(images=pil, return_tensors="pt").to(device)
        with torch.inference_mode():
            embeddings = model.get_image_embeddings(inputs.pixel_values)
        embedding_s = time.perf_counter() - started
        for index, (request, boxes) in enumerate(selected):
            started = time.perf_counter()
            previous = results[request.region_id]
            try:
                coordinates = [[box.x_min, box.y_min, box.x_max, box.y_max] for box, _ in boxes]
                prompts = processor(images=pil, input_boxes=[coordinates], return_tensors="pt").to(device)
                with torch.inference_mode():
                    output = model(image_embeddings=embeddings, input_boxes=prompts.input_boxes, multimask_output=False)
                masks = processor.image_processor.post_process_masks(
                    output.pred_masks.cpu(), prompts.original_sizes.cpu(), prompts.reshaped_input_sizes.cpu(),
                )[0][:, 0].numpy().astype(np.bool_)
                # Stuff pode ter várias localizações independentes. A união
                # envolve somente segmentos já condicionados ao mesmo conceito.
                mask = Mask(np.any(masks, axis=0), image.width, image.height)
                extent = BoundingBox(
                    min(box.x_min for box, _ in boxes), min(box.y_min for box, _ in boxes),
                    max(box.x_max for box, _ in boxes), max(box.y_max for box, _ in boxes),
                )
                results[request.region_id] = replace(
                    previous, model_mask=mask, prompt_box=extent,
                    prompt_boxes=tuple(box for box, _ in boxes),
                    detection_confidence=min(score for _, score in boxes),
                    geometric_confidence=float(output.iou_scores.min().clamp(0, 1).cpu()),
                    status=GroundingStatus.REFINED,
                )
                del prompts, output
            except (RuntimeError, ValueError, TypeError) as error:
                results[request.region_id] = replace(previous, reason=f"Falha de segmentação: {error}")
            results[request.region_id] = replace(
                results[request.region_id],
                model_calls=previous.model_calls + 1 + (1 if index == 0 else 0),
                latency_s=previous.latency_s + time.perf_counter() - started + (embedding_s if index == 0 else 0),
            )
