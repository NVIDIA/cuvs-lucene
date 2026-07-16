#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

MVN_BIN="${MVN:-mvn}"
PYTHON_BIN="${PYTHON:-python3}"
SKIP_BUILD=0
GPU_E2E=0
PYLUCENE_CASES="${CUVS_LUCENE_PYLUCENE_CASES:-}"
PYLUCENE_ROWS="${CUVS_LUCENE_PYLUCENE_ROWS:-}"
PYLUCENE_DIMS="${CUVS_LUCENE_PYLUCENE_DIMS:-}"
PYLUCENE_TOPK="${CUVS_LUCENE_PYLUCENE_TOPK:-}"

for arg in "$@"; do
  case "${arg}" in
    --gpu-e2e)
      GPU_E2E=1
      ;;
    --no-build)
      SKIP_BUILD=1
      ;;
    --cases=*)
      PYLUCENE_CASES="${arg#--cases=}"
      ;;
    --rows=*)
      PYLUCENE_ROWS="${arg#--rows=}"
      ;;
    --dims=*)
      PYLUCENE_DIMS="${arg#--dims=}"
      ;;
    --topk=*)
      PYLUCENE_TOPK="${arg#--topk=}"
      ;;
    -h|--help)
      echo "Usage: $0 [--no-build] [--gpu-e2e] [--cases=CASE[,CASE...]] [--rows=N] [--dims=N] [--topk=N]"
      echo
      echo "Builds and checks the PyLucene sidecar jar, then runs the PyLucene pytest suite."
      echo "Set CUVS_LUCENE_PYLUCENE_JAR to test an existing sidecar jar."
      echo "Set CUVS_LUCENE_CUVS_JAVA_JAR to the base cuvs-java jar if it is not in ~/.m2."
      echo "Set PYTHON or MVN to override the Python or Maven executable."
      echo
      echo "Case groups: gpu-basic, gpu-segments, cpu-hnsw, cagra-hnsw, algorithm-matrix, all."
      echo "Core cases: smoke, hnsw, cagra, hnsw-single, cagra-single."
      echo "Segment cases: hnsw-1seg, cagra-1seg, hnsw-10seg, cagra-10seg,"
      echo "  hnsw-10seg-force-1, cagra-10seg-force-1,"
      echo "  hnsw-100seg-force-10, cagra-100seg-force-10."
      echo "CPU HNSW cases: hnsw-cpu, hnsw-cpu-single, hnsw-cpu-1seg,"
      echo "  hnsw-cpu-10seg, hnsw-cpu-10seg-force-1,"
      echo "  hnsw-cpu-100seg-force-10."
      echo "CAGRA-to-HNSW cases: cagra-hnsw-1layer, cagra-hnsw-3layer."
      echo "--gpu-e2e defaults to cases=all, base rows=2000, dims=32, topk=20."
      echo "High-segment cases use at least 257 rows per segment to avoid expected cuVS graph-degree clamps."
      echo "--gpu-e2e requires a PyLucene environment plus cuVS native support on a GPU machine."
      exit 0
      ;;
    *)
      echo "Unknown argument: ${arg}" >&2
      exit 2
      ;;
  esac
done

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Required command not found: $1" >&2
    exit 127
  fi
}

require_command "${PYTHON_BIN}"
require_command jar

if [[ "${SKIP_BUILD}" -eq 0 && -z "${CUVS_LUCENE_PYLUCENE_JAR:-}" ]]; then
  require_command "${MVN_BIN}"
  "${MVN_BIN}" clean package -DskipTests
fi

project_version="$(
  sed -n 's/.*CUVS_LUCENE#VERSION_UPDATE_MARKER_START--><version>\([^<]*\)<\/version>.*/\1/p' pom.xml \
    | head -n 1
)"
if [[ -z "${project_version}" ]]; then
  echo "Unable to determine project version from pom.xml" >&2
  exit 1
fi

if [[ -n "${CUVS_LUCENE_PYLUCENE_JAR:-}" ]]; then
  sidecar_jar="${CUVS_LUCENE_PYLUCENE_JAR}"
else
  sidecar_jar="target/cuvs-lucene-${project_version}-jar-with-pylucene-dependencies.jar"
fi

if [[ ! -f "${sidecar_jar}" ]]; then
  echo "PyLucene sidecar jar not found: ${sidecar_jar}" >&2
  echo "Run without --no-build, or set CUVS_LUCENE_PYLUCENE_JAR to an existing jar." >&2
  exit 1
fi

case "${sidecar_jar}" in
  /*)
    sidecar_jar_abs="${sidecar_jar}"
    ;;
  *)
    sidecar_jar_abs="${REPO_ROOT}/${sidecar_jar}"
    ;;
esac

find_cuvs_java_jar() {
  if [[ -n "${CUVS_LUCENE_CUVS_JAVA_JAR:-}" ]]; then
    printf '%s\n' "${CUVS_LUCENE_CUVS_JAVA_JAR}"
    return
  fi

  local m2_base="${HOME}/.m2/repository/com/nvidia/cuvs/cuvs-java/${project_version}"
  local jar="${m2_base}/cuvs-java-${project_version}.jar"
  if [[ -f "${jar}" ]]; then
    printf '%s\n' "${jar}"
    return
  fi

  local m2_repo="${HOME}/.m2/repository/com/nvidia/cuvs/cuvs-java"
  if [[ ! -d "${m2_repo}" ]]; then
    return
  fi

  find "${m2_repo}" \
    -type f \
    -name 'cuvs-java-*.jar' \
    ! -name '*sources*' \
    ! -name '*javadoc*' \
    ! -name '*x86_64*' \
    | sort -V \
    | tail -n 1
}

cuvs_java_jar="$(find_cuvs_java_jar)"
if [[ -z "${cuvs_java_jar}" || ! -f "${cuvs_java_jar}" ]]; then
  echo "Base cuvs-java jar not found." >&2
  echo "Set CUVS_LUCENE_CUVS_JAVA_JAR to the base cuvs-java jar, not a native classifier jar." >&2
  exit 1
fi

entries_file="$(mktemp)"
services_dir="$(mktemp -d)"
trap 'rm -f "${entries_file}"; rm -rf "${services_dir}"' EXIT

jar tf "${sidecar_jar_abs}" >"${entries_file}"

for service in \
  "META-INF/services/org.apache.lucene.codecs.Codec" \
  "META-INF/services/org.apache.lucene.codecs.KnnVectorsFormat"; do
  grep -qx "${service}" "${entries_file}" || {
    echo "Missing service descriptor in ${sidecar_jar}: ${service}" >&2
    exit 1
  }
done

for class_file in \
  "com/nvidia/cuvs/lucene/Lucene101AcceleratedHNSWCodec.class" \
  "com/nvidia/cuvs/lucene/Lucene101AcceleratedHNSWBaseLayerCodec.class" \
  "com/nvidia/cuvs/lucene/Lucene101AcceleratedHNSWMultiLayerCodec.class" \
  "com/nvidia/cuvs/lucene/CuVS2510GPUSearchCodec.class" \
  "com/nvidia/cuvs/lucene/LuceneAcceleratedHNSWBinaryQuantizedCodec.class" \
  "com/nvidia/cuvs/lucene/LuceneAcceleratedHNSWScalarQuantizedCodec.class"; do
  grep -qx "${class_file}" "${entries_file}" || {
    echo "Missing cuvs-lucene class in ${sidecar_jar}: ${class_file}" >&2
    exit 1
  }
done

lucene_entries="$(
  grep "^org/apache/lucene/" "${entries_file}" \
    | grep -v '/$' || true
)"
if [[ -n "${lucene_entries}" ]]; then
  echo "${sidecar_jar} contains org.apache.lucene classes; PyLucene must provide Lucene." >&2
  echo "${lucene_entries}" >&2
  exit 1
fi

flattened_cuvs_entries="$(
  grep "^com/nvidia/cuvs/" "${entries_file}" \
    | grep -v '/$' \
    | grep -v "^com/nvidia/cuvs/lucene/" || true
)"
if [[ -n "${flattened_cuvs_entries}" ]]; then
  echo "${sidecar_jar} contains flattened cuvs-java classes; use the base cuvs-java jar separately." >&2
  echo "${flattened_cuvs_entries}" >&2
  exit 1
fi

flattened_multi_release_entries="$(
  grep "^META-INF/versions/.*/com/nvidia/cuvs/" "${entries_file}" \
    | grep -v '/$' || true
)"
if [[ -n "${flattened_multi_release_entries}" ]]; then
  echo "${sidecar_jar} contains flattened multi-release cuvs-java classes." >&2
  echo "${flattened_multi_release_entries}" >&2
  exit 1
fi

extra_lucene_services="$(
  grep "^META-INF/services/org.apache.lucene." "${entries_file}" \
    | grep -v "^META-INF/services/org.apache.lucene.codecs.Codec$" \
    | grep -v "^META-INF/services/org.apache.lucene.codecs.KnnVectorsFormat$" || true
)"
if [[ -n "${extra_lucene_services}" ]]; then
  echo "${sidecar_jar} contains unexpected Lucene service descriptors:" >&2
  echo "${extra_lucene_services}" >&2
  exit 1
fi

(
  cd "${services_dir}"
  jar xf \
    "${sidecar_jar_abs}" \
    META-INF/services/org.apache.lucene.codecs.Codec \
    META-INF/services/org.apache.lucene.codecs.KnnVectorsFormat
)

codec_descriptor="${services_dir}/META-INF/services/org.apache.lucene.codecs.Codec"
format_descriptor="${services_dir}/META-INF/services/org.apache.lucene.codecs.KnnVectorsFormat"

for descriptor in "${codec_descriptor}" "${format_descriptor}"; do
  if grep -q "^org\\.apache\\.lucene\\." "${descriptor}"; then
    echo "${descriptor#"${services_dir}"/} advertises Lucene-owned providers." >&2
    exit 1
  fi
done

for provider in \
  "com.nvidia.cuvs.lucene.Lucene101AcceleratedHNSWCodec" \
  "com.nvidia.cuvs.lucene.Lucene101AcceleratedHNSWBaseLayerCodec" \
  "com.nvidia.cuvs.lucene.Lucene101AcceleratedHNSWMultiLayerCodec" \
  "com.nvidia.cuvs.lucene.CuVS2510GPUSearchCodec" \
  "com.nvidia.cuvs.lucene.LuceneAcceleratedHNSWBinaryQuantizedCodec" \
  "com.nvidia.cuvs.lucene.LuceneAcceleratedHNSWScalarQuantizedCodec"; do
  grep -qx "${provider}" "${codec_descriptor}" || {
    echo "Codec service descriptor missing provider: ${provider}" >&2
    exit 1
  }
done

for provider in \
  "com.nvidia.cuvs.lucene.CuVS2510GPUVectorsFormat" \
  "com.nvidia.cuvs.lucene.Lucene99AcceleratedHNSWVectorsFormat" \
  "com.nvidia.cuvs.lucene.LuceneAcceleratedHNSWBinaryQuantizedVectorsFormat" \
  "com.nvidia.cuvs.lucene.LuceneAcceleratedHNSWScalarQuantizedVectorsFormat"; do
  grep -qx "${provider}" "${format_descriptor}" || {
    echo "KnnVectorsFormat service descriptor missing provider: ${provider}" >&2
    exit 1
  }
done

"${PYTHON_BIN}" -c "import lucene" >/dev/null 2>&1 || {
  echo "Python cannot import PyLucene's lucene module." >&2
  echo "Activate or install a PyLucene environment compatible with this project's Lucene version." >&2
  exit 1
}

"${PYTHON_BIN}" -m pytest --version >/dev/null 2>&1 || {
  echo "Python cannot run pytest." >&2
  echo "Install pytest in the active PyLucene environment." >&2
  exit 1
}

smoke_env=(
  "CUVS_LUCENE_PYLUCENE_JAR=${sidecar_jar_abs}"
  "CUVS_LUCENE_CUVS_JAVA_JAR=${cuvs_java_jar}"
)

if [[ "${GPU_E2E}" -eq 1 ]]; then
  smoke_env+=(
    "CUVS_LUCENE_PYLUCENE_CASES=${PYLUCENE_CASES:-all}"
    "CUVS_LUCENE_PYLUCENE_ROWS=${PYLUCENE_ROWS:-2000}"
    "CUVS_LUCENE_PYLUCENE_DIMS=${PYLUCENE_DIMS:-32}"
    "CUVS_LUCENE_PYLUCENE_TOPK=${PYLUCENE_TOPK:-20}"
    "CUVS_LUCENE_REQUIRE_CUVS=1"
    "CUVS_LUCENE_VERIFY_ALL_CODECS=${CUVS_LUCENE_VERIFY_ALL_CODECS:-1}"
  )
else
  if [[ -n "${PYLUCENE_CASES}" ]]; then
    smoke_env+=("CUVS_LUCENE_PYLUCENE_CASES=${PYLUCENE_CASES}")
  fi
  if [[ -n "${PYLUCENE_ROWS}" ]]; then
    smoke_env+=("CUVS_LUCENE_PYLUCENE_ROWS=${PYLUCENE_ROWS}")
  fi
  if [[ -n "${PYLUCENE_DIMS}" ]]; then
    smoke_env+=("CUVS_LUCENE_PYLUCENE_DIMS=${PYLUCENE_DIMS}")
  fi
  if [[ -n "${PYLUCENE_TOPK}" ]]; then
    smoke_env+=("CUVS_LUCENE_PYLUCENE_TOPK=${PYLUCENE_TOPK}")
  fi
  smoke_env+=("CUVS_LUCENE_VERIFY_ALL_CODECS=${CUVS_LUCENE_VERIFY_ALL_CODECS:-1}")
fi

env "${smoke_env[@]}" "${PYTHON_BIN}" -m pytest -q -s examples/test_pylucene_smoke.py
