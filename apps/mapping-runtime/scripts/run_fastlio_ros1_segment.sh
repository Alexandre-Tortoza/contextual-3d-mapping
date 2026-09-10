#!/usr/bin/env bash

set -euo pipefail

bag_path="${1:?informe o caminho da rosbag dentro do container}"
config_path="${2:?informe o caminho da configuração FAST-LIO}"
duration_seconds="${3:-20}"
output_path="${4:-/workspace/artifacts/fastlio-segment.pcd}"
# Offset de reprodução no relógio de gravação do bag, resolvido por
# `mapping-runtime bag-window`. O bag do corridor-02 preserva dois relógios
# diferentes, e `rosbag play --start` só entende o de gravação.
start_seconds="${5:-0}"
odometry_topic="${6:-/Odometry}"
artifact_directory="$(dirname "${output_path}")"
segment_name="$(basename "${output_path%.pcd}")"
odometry_path="${artifact_directory}/${segment_name}-odometry.csv"

mkdir -p "${artifact_directory}"
source /opt/fast-lio/devel/setup.bash

# Inicia todos os processos no mesmo container para compartilhar o ROS master
# e garantir que o encerramento do mapper persista o PCD acumulado.
roscore >"${artifact_directory}/fastlio-roscore.log" 2>&1 &
core_pid=$!
mapping_pid=""
odometry_pid=""

# Encerra processos auxiliares mesmo quando rosbag ou FAST-LIO falham.
trap 'if [[ -n "${odometry_pid}" ]]; then kill -INT "${odometry_pid}" 2>/dev/null || true; fi; if [[ -n "${mapping_pid}" ]]; then kill -INT "${mapping_pid}" 2>/dev/null || true; fi; kill -INT "${core_pid}" 2>/dev/null || true' EXIT

sleep 2
rosparam load "${config_path}"
rosparam set /feature_extract_enable false
rosparam set /point_filter_num 4
rosparam set /max_iteration 3
rosparam set /filter_size_surf 0.5
rosparam set /filter_size_map 0.5
rosparam set /cube_side_length 1000
rosparam set /runtime_pos_log_enable false

rosrun fast_lio fastlio_mapping >"${artifact_directory}/fastlio-mapping.log" 2>&1 &
mapping_pid=$!
sleep 2

# Registra a trajetória estimada pelo próprio FAST-LIO. Sem ela a associação
# depende do ground-truth do dataset, cuja divergência entra direto na projeção
# quando o contexto é ancorado nos pontos do mapa.
rostopic echo -p "${odometry_topic}" >"${odometry_path}" 2>"${artifact_directory}/fastlio-odometry.log" &
odometry_pid=$!

rosbag play "${bag_path}" --start="${start_seconds}" --duration="${duration_seconds}" \
    >"${artifact_directory}/fastlio-rosbag.log" 2>&1
sleep 2

kill -INT "${odometry_pid}" 2>/dev/null || true
wait "${odometry_pid}" 2>/dev/null || true
odometry_pid=""
kill -INT "${mapping_pid}"
wait "${mapping_pid}" || true
mapping_pid=""

if [[ ! -s "${odometry_path}" ]]; then
    printf 'AVISO: nenhuma odometria capturada em %s (tópico %s)\n' "${odometry_path}" "${odometry_topic}" >&2
fi

generated_pcd=/opt/fast-lio/src/FAST_LIO/PCD/scans.pcd
test -s "${generated_pcd}"
cp "${generated_pcd}" "${output_path}"
printf '%s\n' "${output_path}"
