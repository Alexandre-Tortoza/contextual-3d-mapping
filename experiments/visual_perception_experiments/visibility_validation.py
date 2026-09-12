"""Replay de visibilidade em três braços com percepção e geometria congeladas.

Não executa estimação de pose, reconhecimento nem grounding. IDs de regressão
são entradas de avaliação e nunca são consumidos pela associação.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from mapping_runtime.corridor02_context import Corridor02ContextRequest, export_corridor02_context
from mapping_runtime.keyframe_inputs import resolve_keyframe_inputs


# Persiste resultados progressivos para permitir inspeção mesmo antes do fim.
def _write(path, payload):
    """Grava JSON legível em um diretório de run independente."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')


# Registra os bytes das entradas pequenas; o PCD já é verificado pelo runtime.
def _fingerprint(path):
    """Retorna referência e SHA-256 do arquivo congelado."""
    return {'uri': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


# Compara IDs comuns sem confundir pontos próximos da baseline com ground truth.
def _metrics(payload, regression):
    """Resume labels, perdas próximas e vazamentos nos IDs previamente auditados."""
    by_id = {point['geometry_id']: point for point in payload['points']}
    strong = {key for key, point in by_id.items() if (point.get('context') or {}).get('label') == regression['label']}
    near, far = set(regression['near_ids']), set(regression['far_ids'])
    lost = near - strong
    return {
        'context_summary': payload['context_summary'],
        'label_counts': dict(Counter(point['context']['label'] for point in payload['points'] if (point.get('context') or {}).get('label'))),
        'near_retained': len(near & strong), 'near_lost': len(lost), 'far_retained': len(far & strong),
        'additional_label_points': len(strong - near - far),
        'lost_near_states': dict(Counter((by_id[key].get('context') or {}).get('semantic_status') or (by_id[key].get('context') or {}).get('status', 'missing') for key in lost)),
        'lost_near': [{'geometry_id': key, 'coordinates_m': by_id[key]['coordinates_m'], 'context': by_id[key].get('context')} for key in sorted(lost)],
        'far_points': [{'geometry_id': key, 'coordinates_m': by_id[key]['coordinates_m'], 'context': by_id[key].get('context')} for key in sorted(far)],
        'visibility_counts': dict(sum((Counter(obs.get('visibility_counts',{})) for obs in payload['observations']),Counter())),
        'region_support_counts': dict(Counter((region.get('surface_support') or {}).get('status','not_evaluated') for region in payload['regions'])),
    }


# Torna os três resultados comparáveis visualmente na mesma escala do mapa.
def _plot(destination, paths, regression):
    """Grava vista superior comum e comparação dos pontos próximos do palete."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(2, len(paths), figsize=(16,8), layout='constrained', squeeze=False)
    near = set(regression['near_ids'])
    for column, (name, path) in enumerate(paths.items()):
        payload=json.loads(path.read_text())
        geometry=np.array([p['coordinates_m'] for p in payload['points']])
        labelled=np.array([p['coordinates_m'] for p in payload['points'] if (p.get('context')or{}).get('label')==regression['label']]).reshape(-1,3)
        reference=np.array([p['coordinates_m'] for p in payload['points'] if p['geometry_id'] in near])
        axes[0,column].scatter(geometry[::5,0],geometry[::5,1],s=.2,c='#aeb4bc')
        axes[0,column].scatter(labelled[:,0],labelled[:,1],s=6,c='#ed7820')
        axes[0,column].set(xlim=(5,85),ylim=(-43,7),title=name,xlabel='x do mapa (m)',ylabel='y do mapa (m)')
        axes[1,column].scatter(reference[:,0],reference[:,2],s=10,c='#bdc2cb',label='Pontos próximos na baseline')
        axes[1,column].scatter(labelled[:,0],labelled[:,2],s=8,c='#ed7820',label='Rótulo forte nesta run')
        axes[1,column].set(xlim=(9.8,12),ylim=(-1.2,.6),xlabel='x do mapa (m)',ylabel='z do mapa (m)')
        axes[1,column].legend(fontsize=7)
    fig.suptitle('Associação do palete · geometria e máscaras congeladas')
    fig.savefig(destination/'comparison.png',dpi=150)
    plt.close(fig)


# Orquestra apenas associação/fusão e conserva todas as entradas para reprodução.
def main(argv=None):
    """Executa as ablações e grava métricas e proveniência sem substituir runs."""
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('geometry','bag','intrinsics','extrinsics','odometry','window','visual-run','baseline','regression-ids','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    options=parser.parse_args(argv)
    destination=options.output
    destination.mkdir(parents=True,exist_ok=False)
    frames, missing, anchor=resolve_keyframe_inputs(options.window,options.visual_run)
    if missing:
        raise ValueError(f'Percepção congelada incompleta: {missing}')
    regression=json.loads(options.regression_ids.read_text())
    baseline=json.loads(options.baseline.read_text())
    if baseline['source']['sha256'] != regression['geometry_sha256']:
        raise ValueError('Os IDs de regressão pertencem a outra geometria.')
    manifest={'status':'running','frame_count':len(frames),'inputs':{name:_fingerprint(getattr(options,name)) for name in ('geometry','intrinsics','extrinsics','odometry','window','baseline','regression_ids')},
              'bag':str(options.bag.resolve()),'visual_inputs':[_fingerprint(frame.visual_observation) for frame in frames],
              'image_inputs':[_fingerprint(path) for frame in frames for path in (frame.raw_image,frame.valid_area_mask,*((frame.ego_mask,) if frame.ego_mask else ()))],
              'arms':{}}
    manifest['command'] = sys.argv
    manifest['git_head'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    manifest['implementation'] = [_fingerprint(path) for pattern in (
        'modules/sensor-association/src/sensor_association/*.py',
        'modules/geometric-map/src/geometric_map/*.py',
        'apps/mapping-runtime/src/mapping_runtime/corridor02_context.py',
        'experiments/visual_perception_experiments/visibility_validation.py',
    ) for path in sorted(Path('.').glob(pattern))]
    for item in manifest['implementation']:
        source = Path(item['uri'])
        snapshot = destination / 'code' / source.relative_to(Path.cwd())
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, snapshot)
    manifest['implementation_snapshot'] = 'code'
    _write(destination/'manifest.json',manifest)
    shutil.copyfile(options.window,destination/'window.json')
    shutil.copyfile(options.regression_ids,destination/'regression-ids.json')
    metrics={'frozen_baseline':_metrics(baseline,regression)}
    paths={}
    for mode in ('legacy_cells','dense_cells','measured_surfaces'):
        output=destination/mode/'context.json'
        print(f'Compondo {mode}: {len(frames)} frames congelados',flush=True)
        started=time.monotonic()
        request=Corridor02ContextRequest(options.geometry,options.bag,options.intrinsics,options.extrinsics,frames,output,
            odometry=options.odometry,pose_anchor_ns=anchor,visibility_mode=mode,
            audit_geometry_ids=frozenset(regression['near_ids'] + regression['far_ids']))
        export_corridor02_context(request)
        payload=json.loads(output.read_text())
        paths[mode]=output
        metrics[mode]=_metrics(payload,regression)
        manifest['arms'][mode]={'artifact':str(output.resolve()),'elapsed_s':time.monotonic()-started,
            'surface_config':asdict(request.surface_config) if mode=='measured_surfaces' else None,
            'sha256':_fingerprint(output)['sha256']}
        _write(destination/'metrics.json',metrics)
        _write(destination/'manifest.json',manifest)
        print(mode,{key:value for key,value in metrics[mode].items() if key in ('label_counts','near_retained','near_lost','far_retained','additional_label_points')},flush=True)
    _plot(destination,paths,regression)
    # A baseline recomposta deve preservar os IDs fortes, não apenas a contagem.
    recomposed=json.loads(paths['legacy_cells'].read_text())
    baseline_labels={p['geometry_id']:(p.get('context')or{}).get('label') for p in baseline['points']}
    new_labels={p['geometry_id']:(p.get('context')or{}).get('label') for p in recomposed['points']}
    manifest['baseline_labels_identical']=baseline_labels==new_labels
    manifest['far_regression_passed']=metrics['measured_surfaces']['far_retained']==0
    manifest['status']='complete'
    _write(destination/'manifest.json',manifest)
    if not manifest['baseline_labels_identical'] or not manifest['far_regression_passed']:
        raise AssertionError('A comparação não satisfez a baseline ou a regressão de pontos distantes.')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
