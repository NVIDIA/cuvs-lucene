# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path

import lucene


REPO_ROOT = Path(__file__).resolve().parents[1]
HNSW_CODEC = "Lucene101AcceleratedHNSWCodec"
CAGRA_HNSW_BASE_LAYER_CODEC = "Lucene101AcceleratedHNSWBaseLayerCodec"
CAGRA_HNSW_MULTI_LAYER_CODEC = "Lucene101AcceleratedHNSWMultiLayerCodec"
CAGRA_CODEC = "CuVS2510GPUSearchCodec"
HNSW_SEARCH_CODECS = (
    HNSW_CODEC,
    CAGRA_HNSW_BASE_LAYER_CODEC,
    CAGRA_HNSW_MULTI_LAYER_CODEC,
)
EXPECTED_CODECS = (
    HNSW_CODEC,
    CAGRA_HNSW_BASE_LAYER_CODEC,
    CAGRA_HNSW_MULTI_LAYER_CODEC,
    CAGRA_CODEC,
    "Lucene101AcceleratedHNSWBinaryQuantizedCodec",
    "Lucene101AcceleratedHNSWScalarQuantizedCodec",
)
BASIC_GPU_CASES = ("hnsw", "cagra", "hnsw-single", "cagra-single")
SEGMENT_GPU_CASES = (
    "hnsw-1seg",
    "cagra-1seg",
    "hnsw-10seg",
    "cagra-10seg",
    "hnsw-10seg-force-1",
    "cagra-10seg-force-1",
    "hnsw-100seg-force-10",
    "cagra-100seg-force-10",
)
CPU_HNSW_CASES = (
    "hnsw-cpu",
    "hnsw-cpu-single",
    "hnsw-cpu-1seg",
    "hnsw-cpu-10seg",
    "hnsw-cpu-10seg-force-1",
    "hnsw-cpu-100seg-force-10",
)
CAGRA_HNSW_CASES = (
    "cagra-hnsw-1layer",
    "cagra-hnsw-3layer",
)
ALGORITHM_MATRIX_CASES = (
    "hnsw-cpu",
    "cagra-hnsw-1layer",
    "cagra-hnsw-3layer",
    "cagra",
)
CASE_GROUPS = {
    "gpu": BASIC_GPU_CASES,
    "gpu-basic": BASIC_GPU_CASES,
    "gpu-segments": SEGMENT_GPU_CASES,
    "segments": SEGMENT_GPU_CASES,
    "cpu-hnsw": CPU_HNSW_CASES,
    "cagra-hnsw": CAGRA_HNSW_CASES,
    "algorithm-matrix": ALGORITHM_MATRIX_CASES,
    "all": BASIC_GPU_CASES + SEGMENT_GPU_CASES + CPU_HNSW_CASES + CAGRA_HNSW_CASES,
}
ID_FIELD = "id"
GROUP_FIELD = "group"
VECTOR_FIELD = "vector"
MIN_ROWS_PER_SEGMENT_WITH_DEFAULT_GRAPH_PARAMS = 257
# graphDegree=32 gives M=16, and the smoke suite omits every 11th vector when
# update coverage is enabled. Keep the second upper HNSW layer above cuVS's
# derived NN_DESCENT intermediate degree of 96 to avoid native clamp warnings.
MIN_ROWS_FOR_THREE_HNSW_LAYERS_WITH_GRAPH_DEGREE_32 = 27316
FORCE_CPU_HNSW_FALLBACK_PROPERTY = "cuvs.lucene.forceCpuHnswFallback"


@dataclass(frozen=True)
class SmokeCase:
    name: str
    codec_name: str
    row_count: int
    dims: int
    top_k: int
    expected_suffixes: tuple
    require_cuvs: bool = False
    exercise_updates: bool = False
    segment_count: int = 1
    force_merge_target: int = 0
    disable_background_merges: bool = False
    assert_segments: bool = False
    force_cpu_hnsw: bool = False
    expected_hnsw_layers: int = 0
    expected_cagra_graph_build_algo: str = ""
    expected_cagra_graph_degree: int = 0
    expected_cagra_intermediate_graph_degree: int = 0


@dataclass
class PyLuceneContext:
    cuvs_lucene_jar: Path
    cuvs_java_jar: Path
    codec_class: object
    jarray: object
    codec_cache: dict


def env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def int_env(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return int(value)


def suffixes_from_env():
    configured = os.environ.get("CUVS_LUCENE_EXPECT_INDEX_SUFFIXES")
    if configured:
        return tuple(suffix.strip() for suffix in configured.split(",") if suffix.strip())
    if os.environ.get("CUVS_LUCENE_EXPECT_CUVS_FILES") == "1":
        return (".vcag", ".vemc")
    return ()


def find_cuvs_lucene_jar():
    configured = os.environ.get("CUVS_LUCENE_PYLUCENE_JAR") or os.environ.get(
        "CUVS_LUCENE_JAR"
    )
    if configured:
        jar = Path(configured)
        if not jar.exists():
            raise FileNotFoundError(f"Configured cuvs-lucene jar does not exist: {jar}")
        return jar

    jars = sorted(
        (REPO_ROOT / "target").glob(
            "cuvs-lucene-*-jar-with-pylucene-dependencies.jar"
        )
    )
    if not jars:
        raise FileNotFoundError(
            "No PyLucene sidecar jar found under target/. "
            "Run `mvn clean package -DskipTests` first."
        )
    return jars[-1]


def find_cuvs_java_jar():
    configured = os.environ.get("CUVS_LUCENE_CUVS_JAVA_JAR") or os.environ.get(
        "CUVS_JAVA_JAR"
    )
    if configured:
        jar = Path(configured)
        if not jar.exists():
            raise FileNotFoundError(f"Configured cuvs-java jar does not exist: {jar}")
        return jar

    m2_repo = Path.home() / ".m2" / "repository" / "com" / "nvidia" / "cuvs" / "cuvs-java"
    if not m2_repo.exists():
        raise FileNotFoundError(
            "Unable to find cuvs-java in ~/.m2. Set CUVS_LUCENE_CUVS_JAVA_JAR "
            "to the base cuvs-java jar, not a native classifier jar."
        )

    def is_base_cuvs_java_jar(jar):
        return (
            jar.name.startswith("cuvs-java-")
            and jar.name.endswith(".jar")
            and "-x86_64-" not in jar.name
            and "-sources" not in jar.name
            and "-javadoc" not in jar.name
        )

    jars = sorted(
        jar for jar in m2_repo.glob("*/*.jar") if is_base_cuvs_java_jar(jar)
    )
    if not jars:
        raise FileNotFoundError(
            "Unable to find the base cuvs-java jar in ~/.m2. Set "
            "CUVS_LUCENE_CUVS_JAVA_JAR explicitly."
        )
    return jars[-1]


def vector_for(doc_id, dims):
    x = ((doc_id + 1) * 2654435761) & 0xFFFFFFFF
    values = []
    for dim in range(dims):
        x = (1664525 * x + 1013904223 + dim * 17) & 0xFFFFFFFF
        values.append((x / 4294967295.0) * 2.0 - 1.0)
    return values


def fvec(jarray, values):
    return jarray("float")([float(value) for value in values])


def init_vm(cuvs_java_jar, cuvs_lucene_jar):
    java_library_path = os.environ.get("JAVA_LIBRARY_PATH") or os.environ.get(
        "LD_LIBRARY_PATH"
    )
    vmargs = [
        "--enable-native-access=ALL-UNNAMED",
        "--add-modules=jdk.incubator.vector",
    ]
    if java_library_path:
        vmargs.append(f"-Djava.library.path={java_library_path}")

    lucene.initVM(
        classpath=os.pathsep.join(
            [str(cuvs_java_jar), str(cuvs_lucene_jar), lucene.CLASSPATH]
        ),
        vmargs=vmargs,
    )


def quiet_expected_provider_fallback_logs():
    if not env_flag("CUVS_LUCENE_QUIET_EXPECTED_PROVIDER_LOGS", True):
        return
    try:
        from java.util.logging import Level, Logger

        Logger.getLogger("com.nvidia.cuvs.lucene.LuceneProvider").setLevel(Level.OFF)
        Logger.getLogger(
            "com.nvidia.cuvs.lucene.Lucene99AcceleratedHNSWVectorsFormat"
        ).setLevel(Level.SEVERE)
        Logger.getLogger(
            "org.apache.lucene.internal.vectorization.PanamaVectorizationProvider"
        ).setLevel(Level.WARNING)
    except Exception:
        pass


def print_threadlocal_provider_diagnostics():
    if not env_flag("CUVS_LUCENE_PRINT_THREADLOCAL_PROVIDER", False):
        return

    from java.lang import System

    print("ThreadLocalCuVSResourcesProvider diagnostics:")
    value = System.getProperty(FORCE_CPU_HNSW_FALLBACK_PROPERTY)
    print(
        f"  {FORCE_CPU_HNSW_FALLBACK_PROPERTY}="
        f"{value if value is not None else '<unset>'}"
    )

    source_path = (
        REPO_ROOT
        / "src"
        / "main"
        / "java"
        / "com"
        / "nvidia"
        / "cuvs"
        / "lucene"
        / "ThreadLocalCuVSResourcesProvider.java"
    )
    if not source_path.exists():
        print(f"  source={source_path} (missing)")
        return

    print(f"  source={source_path}")
    for line_number, line in enumerate(source_path.read_text().splitlines(), start=1):
        print(f"  {line_number:4d}: {line}")


def expand_case_names(names):
    expanded = []
    for name in names:
        group = CASE_GROUPS.get(name)
        if group:
            expanded.extend(expand_case_names(group))
        else:
            expanded.append(name)
    return expanded


def matrix_case(
    name,
    codec_name,
    expected_suffixes,
    require_cuvs,
    segment_count=1,
    force_merge_target=0,
    disable_background_merges=False,
    assert_segments=False,
    exercise_updates=False,
    row_count_floor=0,
    force_cpu_hnsw=False,
    expected_hnsw_layers=0,
    expected_cagra_graph_build_algo="",
    expected_cagra_graph_degree=0,
    expected_cagra_intermediate_graph_degree=0,
):
    matrix_rows = max(int_env("CUVS_LUCENE_PYLUCENE_ROWS", 2000), row_count_floor)
    matrix_dims = int_env("CUVS_LUCENE_PYLUCENE_DIMS", 32)
    matrix_top_k = int_env("CUVS_LUCENE_PYLUCENE_TOPK", 20)
    return SmokeCase(
        name=name,
        codec_name=codec_name,
        row_count=matrix_rows,
        dims=matrix_dims,
        top_k=matrix_top_k,
        expected_suffixes=expected_suffixes,
        require_cuvs=require_cuvs,
        exercise_updates=exercise_updates,
        segment_count=segment_count,
        force_merge_target=force_merge_target,
        disable_background_merges=disable_background_merges,
        assert_segments=assert_segments,
        force_cpu_hnsw=force_cpu_hnsw,
        expected_hnsw_layers=expected_hnsw_layers,
        expected_cagra_graph_build_algo=expected_cagra_graph_build_algo,
        expected_cagra_graph_degree=expected_cagra_graph_degree,
        expected_cagra_intermediate_graph_degree=expected_cagra_intermediate_graph_degree,
    )


def build_segment_case(
    name,
    codec_name,
    expected_suffixes,
    require_cuvs,
    segments,
    force_target,
    min_rows_per_segment=MIN_ROWS_PER_SEGMENT_WITH_DEFAULT_GRAPH_PARAMS,
    force_cpu_hnsw=False,
    expected_hnsw_layers=0,
    expected_cagra_graph_build_algo="",
    expected_cagra_graph_degree=0,
    expected_cagra_intermediate_graph_degree=0,
):
    force_merge_target = force_target or 0
    return matrix_case(
        name=name,
        codec_name=codec_name,
        expected_suffixes=expected_suffixes,
        require_cuvs=require_cuvs,
        segment_count=segments,
        force_merge_target=force_merge_target,
        disable_background_merges=(force_merge_target == 0),
        assert_segments=True,
        row_count_floor=segments * min_rows_per_segment,
        force_cpu_hnsw=force_cpu_hnsw,
        expected_hnsw_layers=expected_hnsw_layers,
        expected_cagra_graph_build_algo=expected_cagra_graph_build_algo,
        expected_cagra_graph_degree=expected_cagra_graph_degree,
        expected_cagra_intermediate_graph_degree=expected_cagra_intermediate_graph_degree,
    )


def build_named_case(name):
    matrix_dims = int_env("CUVS_LUCENE_PYLUCENE_DIMS", 32)
    force_merge = env_flag("CUVS_LUCENE_PYLUCENE_FORCE_MERGE", True)
    exercise_updates = env_flag("CUVS_LUCENE_PYLUCENE_EXERCISE_UPDATES", True)
    require_cuvs = env_flag("CUVS_LUCENE_REQUIRE_CUVS", False)

    if name == "smoke":
        return SmokeCase(
            name="smoke",
            codec_name=HNSW_CODEC,
            row_count=int_env("CUVS_LUCENE_PYLUCENE_ROWS", 2),
            dims=int_env("CUVS_LUCENE_PYLUCENE_DIMS", 3),
            top_k=int_env("CUVS_LUCENE_PYLUCENE_TOPK", 2),
            expected_suffixes=suffixes_from_env(),
            require_cuvs=require_cuvs,
        )
    if name == "hnsw":
        return matrix_case(
            name="hnsw",
            codec_name=HNSW_CODEC,
            expected_suffixes=(".vex", ".vem"),
            require_cuvs=require_cuvs,
            exercise_updates=exercise_updates,
            segment_count=3,
            force_merge_target=(1 if force_merge else 0),
        )
    if name == "cagra":
        return matrix_case(
            name="cagra",
            codec_name=CAGRA_CODEC,
            expected_suffixes=(".vcag", ".vemc"),
            require_cuvs=True,
            exercise_updates=exercise_updates,
            segment_count=3,
            force_merge_target=(1 if force_merge else 0),
        )
    if name in {"hnsw-single", "single-hnsw"}:
        return SmokeCase(
            name="hnsw-single",
            codec_name=HNSW_CODEC,
            row_count=1,
            dims=matrix_dims,
            top_k=1,
            expected_suffixes=(".vex", ".vem"),
            require_cuvs=require_cuvs,
        )
    if name in {"cagra-single", "single-cagra"}:
        return SmokeCase(
            name="cagra-single",
            codec_name=CAGRA_CODEC,
            row_count=1,
            dims=matrix_dims,
            top_k=1,
            expected_suffixes=(".vcag", ".vemc"),
            require_cuvs=True,
        )
    if name == "hnsw-1seg":
        return build_segment_case(name, HNSW_CODEC, (".vex", ".vem"), require_cuvs, 1, 0)
    if name == "cagra-1seg":
        return build_segment_case(name, CAGRA_CODEC, (".vcag", ".vemc"), True, 1, 0)
    if name == "hnsw-10seg":
        return build_segment_case(name, HNSW_CODEC, (".vex", ".vem"), require_cuvs, 10, 0)
    if name == "cagra-10seg":
        return build_segment_case(name, CAGRA_CODEC, (".vcag", ".vemc"), True, 10, 0)
    if name == "hnsw-10seg-force-1":
        return build_segment_case(name, HNSW_CODEC, (".vex", ".vem"), require_cuvs, 10, 1)
    if name == "cagra-10seg-force-1":
        return build_segment_case(name, CAGRA_CODEC, (".vcag", ".vemc"), True, 10, 1)
    if name == "hnsw-100seg-force-10":
        return build_segment_case(name, HNSW_CODEC, (".vex", ".vem"), require_cuvs, 100, 10)
    if name == "cagra-100seg-force-10":
        return build_segment_case(name, CAGRA_CODEC, (".vcag", ".vemc"), True, 100, 10)
    if name in {"cagra-hnsw-1layer", "cagra-hnsw-base", "cagra-hnsw-base-layer"}:
        return matrix_case(
            name="cagra-hnsw-1layer",
            codec_name=CAGRA_HNSW_BASE_LAYER_CODEC,
            expected_suffixes=(".vex", ".vem"),
            require_cuvs=True,
            exercise_updates=exercise_updates,
            segment_count=1,
            expected_hnsw_layers=1,
            expected_cagra_graph_build_algo="NN_DESCENT",
            expected_cagra_graph_degree=32,
            expected_cagra_intermediate_graph_degree=64,
        )
    if name in {
        "cagra-hnsw-3layer",
        "cagra-hnsw-multilayer",
        "cagra-hnsw-multi",
        "cagra-hnsw-multi-layer",
    }:
        return matrix_case(
            name="cagra-hnsw-3layer",
            codec_name=CAGRA_HNSW_MULTI_LAYER_CODEC,
            expected_suffixes=(".vex", ".vem"),
            require_cuvs=True,
            exercise_updates=exercise_updates,
            segment_count=1,
            row_count_floor=MIN_ROWS_FOR_THREE_HNSW_LAYERS_WITH_GRAPH_DEGREE_32,
            expected_hnsw_layers=3,
            expected_cagra_graph_build_algo="NN_DESCENT",
            expected_cagra_graph_degree=32,
            expected_cagra_intermediate_graph_degree=64,
        )
    if name == "hnsw-cpu":
        return matrix_case(
            name="hnsw-cpu",
            codec_name=HNSW_CODEC,
            expected_suffixes=(".vex", ".vem"),
            require_cuvs=False,
            exercise_updates=exercise_updates,
            segment_count=3,
            force_merge_target=(1 if force_merge else 0),
            force_cpu_hnsw=True,
        )
    if name in {"hnsw-cpu-single", "single-hnsw-cpu"}:
        return SmokeCase(
            name="hnsw-cpu-single",
            codec_name=HNSW_CODEC,
            row_count=1,
            dims=matrix_dims,
            top_k=1,
            expected_suffixes=(".vex", ".vem"),
            force_cpu_hnsw=True,
        )
    if name == "hnsw-cpu-1seg":
        return build_segment_case(
            name, HNSW_CODEC, (".vex", ".vem"), False, 1, 0, force_cpu_hnsw=True
        )
    if name == "hnsw-cpu-10seg":
        return build_segment_case(
            name, HNSW_CODEC, (".vex", ".vem"), False, 10, 0, force_cpu_hnsw=True
        )
    if name == "hnsw-cpu-10seg-force-1":
        return build_segment_case(
            name, HNSW_CODEC, (".vex", ".vem"), False, 10, 1, force_cpu_hnsw=True
        )
    if name == "hnsw-cpu-100seg-force-10":
        return build_segment_case(
            name, HNSW_CODEC, (".vex", ".vem"), False, 100, 10, force_cpu_hnsw=True
        )
    raise ValueError(f"Unknown PyLucene smoke case: {name}")


def cases_from_env():
    configured_cases = os.environ.get("CUVS_LUCENE_PYLUCENE_CASES")
    if configured_cases:
        names = [name.strip() for name in configured_cases.split(",") if name.strip()]
        return [build_named_case(name) for name in expand_case_names(names)]

    configured_codec = os.environ.get("CUVS_LUCENE_PYLUCENE_CODEC")
    if configured_codec:
        return [
            SmokeCase(
                name="custom",
                codec_name=configured_codec,
                row_count=int_env("CUVS_LUCENE_PYLUCENE_ROWS", 2),
                dims=int_env("CUVS_LUCENE_PYLUCENE_DIMS", 3),
                top_k=int_env("CUVS_LUCENE_PYLUCENE_TOPK", 2),
                expected_suffixes=suffixes_from_env(),
                require_cuvs=env_flag("CUVS_LUCENE_REQUIRE_CUVS", False),
                exercise_updates=env_flag("CUVS_LUCENE_PYLUCENE_EXERCISE_UPDATES", False),
                force_merge_target=(
                    1 if env_flag("CUVS_LUCENE_PYLUCENE_FORCE_MERGE", False) else 0
                ),
            )
        ]

    return [build_named_case("smoke")]


SELECTED_CASES = cases_from_env()


def verify_codecs_advertised(codec_class):
    available_codecs = codec_class.availableCodecs()
    codecs_to_check = (EXPECTED_CODECS if env_flag("CUVS_LUCENE_VERIFY_ALL_CODECS", True) else ())
    for codec_name in codecs_to_check:
        if not available_codecs.contains(codec_name):
            raise AssertionError(
                f"{codec_name} was not advertised by Lucene SPI. "
                f"Available codecs: {available_codecs}"
            )


def writer_telemetry(codec):
    vector_format = codec.knnVectorsFormat()
    description = str(vector_format)
    _, separator, payload = description.partition("(")
    if not separator or not payload.endswith(")"):
        raise AssertionError(f"Malformed vector format diagnostics: {description!r}")
    payload = payload[:-1]
    telemetry = {}
    for item in payload.split(";"):
        key, separator, value = item.partition("=")
        if not separator or not key:
            raise AssertionError(f"Malformed writer telemetry item: {item!r}")
        telemetry[key] = value
    return telemetry


def expected_writer_path(case):
    if case.codec_name == CAGRA_CODEC:
        return "gpu-cagra"
    if case.force_cpu_hnsw:
        return "cpu-hnsw-fallback"
    if case.codec_name in HNSW_SEARCH_CODECS and case.require_cuvs:
        return "gpu-hnsw"
    return None


def observed_writer_path(case, telemetry):
    observed = telemetry.get("writerPath")
    expected = expected_writer_path(case)
    if expected is not None and observed != expected:
        raise AssertionError(
            f"{case.name}: expected writer path {expected}, got {observed or '<unset>'}"
        )
    return observed or "unknown"


def assert_hnsw_telemetry(case, telemetry):
    if case.expected_hnsw_layers:
        observed_layers = telemetry.get("hnswLayers")
        expected_layers = str(case.expected_hnsw_layers)
        if observed_layers != expected_layers:
            raise AssertionError(
                f"{case.name}: expected HNSW layers {expected_layers}, "
                f"got {observed_layers or '<unset>'}"
            )

    if case.expected_cagra_graph_build_algo:
        observed_algo = telemetry.get("cagraGraphBuildAlgo")
        if observed_algo != case.expected_cagra_graph_build_algo:
            raise AssertionError(
                f"{case.name}: expected CAGRA graph build algorithm "
                f"{case.expected_cagra_graph_build_algo}, got {observed_algo or '<unset>'}"
            )

    if case.expected_cagra_graph_degree:
        observed_degree = telemetry.get("cagraGraphDegree")
        expected_degree = str(case.expected_cagra_graph_degree)
        if observed_degree != expected_degree:
            raise AssertionError(
                f"{case.name}: expected CAGRA graph degree {expected_degree}, "
                f"got {observed_degree or '<unset>'}"
            )

    if case.expected_cagra_intermediate_graph_degree:
        observed_intermediate_degree = telemetry.get("cagraIntermediateGraphDegree")
        expected_intermediate_degree = str(case.expected_cagra_intermediate_graph_degree)
        if observed_intermediate_degree != expected_intermediate_degree:
            raise AssertionError(
                f"{case.name}: expected CAGRA intermediate graph degree "
                f"{expected_intermediate_degree}, "
                f"got {observed_intermediate_degree or '<unset>'}"
            )


def build_search_label(case, writer_path):
    if case.codec_name == CAGRA_CODEC:
        label = "build=cagra, search=cagra"
    elif case.force_cpu_hnsw or writer_path.startswith("cpu-hnsw"):
        label = "build=hnsw, search=hnsw"
    elif case.codec_name in HNSW_SEARCH_CODECS:
        label = "build=cagra, search=hnsw"
    else:
        label = "build=unknown, search=unknown"

    if case.expected_hnsw_layers:
        label += f", hnswLayers={case.expected_hnsw_layers}"
    if case.expected_cagra_graph_build_algo:
        label += f", cagraGraphBuildAlgo={case.expected_cagra_graph_build_algo}"
    if case.expected_cagra_graph_degree:
        label += f", cagraGraphDegree={case.expected_cagra_graph_degree}"
    if case.expected_cagra_intermediate_graph_degree:
        label += (
            ", cagraIntermediateGraphDegree="
            f"{case.expected_cagra_intermediate_graph_degree}"
        )
    return label


@contextmanager
def hnsw_cpu_fallback(case):
    if not case.force_cpu_hnsw:
        yield
        return

    from java.lang import System

    previous_value = System.getProperty(FORCE_CPU_HNSW_FALLBACK_PROPERTY)
    System.setProperty(FORCE_CPU_HNSW_FALLBACK_PROPERTY, "true")
    try:
        yield
    finally:
        if previous_value is None:
            System.clearProperty(FORCE_CPU_HNSW_FALLBACK_PROPERTY)
        else:
            System.setProperty(FORCE_CPU_HNSW_FALLBACK_PROPERTY, previous_value)


def missing_doc_ids(case):
    if not case.exercise_updates or case.row_count < 30:
        return set()
    return {doc_id for doc_id in range(case.row_count) if doc_id % 11 == 0}


def deleted_doc_ids(case, missing_ids):
    if not case.exercise_updates or case.row_count < 30:
        return set()

    deleted = set()
    for target in (case.row_count // 4, case.row_count // 2, (case.row_count * 3) // 4):
        for offset in range(case.row_count):
            for candidate in (target + offset, target - offset):
                if 0 <= candidate < case.row_count and candidate not in missing_ids:
                    deleted.add(candidate)
                    break
            if len(deleted) >= 3:
                break
        if len(deleted) >= 3:
            break
    return deleted


def choose_query_ids(row_count, active_vector_ids):
    targets = [0, row_count // 2, row_count - 1]
    query_ids = []
    for target in targets:
        nearest = min(active_vector_ids, key=lambda doc_id: abs(doc_id - target))
        if nearest not in query_ids:
            query_ids.append(nearest)
    return query_ids


def effective_segment_count(case):
    return max(1, min(case.segment_count, case.row_count))


def segment_end_doc_ids(case):
    segments = effective_segment_count(case)
    docs_per_segment, remainder = divmod(case.row_count, segments)
    end_doc_ids = set()
    end_exclusive = 0
    for segment_id in range(segments):
        end_exclusive += docs_per_segment + (1 if segment_id < remainder else 0)
        end_doc_ids.add(end_exclusive - 1)
    return end_doc_ids


def assert_expected_index_files(index_path, expected_suffixes):
    if not expected_suffixes:
        return
    index_files = sorted(path.name for path in Path(index_path).iterdir())
    for suffix in expected_suffixes:
        if not any(name.endswith(suffix) for name in index_files):
            raise AssertionError(
                f"No index file ending with {suffix} found: {index_files}"
            )


def assert_vector_metadata(reader, vector_field, expected_count, expected_dims):
    vector_count = 0
    for leaf_reader_context in reader.leaves():
        leaf_reader = leaf_reader_context.reader()
        values = leaf_reader.getFloatVectorValues(vector_field)
        if values is None:
            continue
        if values.dimension() != expected_dims:
            raise AssertionError(
                f"Vector dimension mismatch: expected {expected_dims}, got {values.dimension()}"
            )
        vector_count += values.size()
    if vector_count != expected_count:
        raise AssertionError(
            f"Vector count mismatch: expected {expected_count}, got {vector_count}"
        )


def assert_segment_topology(reader, case):
    if not case.assert_segments:
        return

    segment_count = sum(1 for _ in reader.leaves())
    if case.force_merge_target:
        if segment_count > case.force_merge_target:
            raise AssertionError(
                f"{case.name}: expected at most {case.force_merge_target} segment(s), "
                f"got {segment_count}"
            )
        if case.force_merge_target == 1 and segment_count != 1:
            raise AssertionError(f"{case.name}: expected one merged segment, got {segment_count}")
        return

    if case.disable_background_merges:
        expected_segments = effective_segment_count(case)
        if segment_count != expected_segments:
            raise AssertionError(
                f"{case.name}: expected {expected_segments} unmerged segment(s), "
                f"got {segment_count}"
            )


def hit_ids(stored_fields, hits):
    return [stored_fields.document(hit.doc).get(ID_FIELD) for hit in hits]


def assert_search_results(searcher, jarray, case, query_ids, inactive_ids):
    from org.apache.lucene.search import KnnFloatVectorQuery

    stored_fields = searcher.storedFields()
    top_k = min(case.top_k, max(1, case.row_count - len(inactive_ids)))
    for query_id in query_ids:
        query = KnnFloatVectorQuery(
            VECTOR_FIELD, fvec(jarray, vector_for(query_id, case.dims)), top_k
        )
        ids = hit_ids(stored_fields, searcher.search(query, top_k).scoreDocs)
        expected = f"doc-{query_id}"
        if expected not in ids:
            raise AssertionError(f"{case.name}: expected {expected} in top {top_k}, got {ids}")
        bad_ids = [doc_id for doc_id in ids if doc_id in inactive_ids]
        if bad_ids:
            raise AssertionError(f"{case.name}: inactive docs returned: {bad_ids}")


def assert_filtered_search(searcher, jarray, case, query_id):
    from org.apache.lucene.index import Term
    from org.apache.lucene.search import KnnFloatVectorQuery, TermQuery

    expected = f"doc-{query_id}"
    filter_query = TermQuery(Term(ID_FIELD, expected))
    query = KnnFloatVectorQuery(
        VECTOR_FIELD, fvec(jarray, vector_for(query_id, case.dims)), 1, filter_query
    )
    ids = hit_ids(searcher.storedFields(), searcher.search(query, 1).scoreDocs)
    if ids != [expected]:
        raise AssertionError(f"{case.name}: filtered search expected {[expected]}, got {ids}")


def codec_for_case(case, codec_class, codec_cache):
    codec = codec_cache.get(case.codec_name)
    if codec is not None:
        return codec

    codec = codec_class.forName(case.codec_name)
    if codec.getName() != case.codec_name:
        raise AssertionError(f"Expected codec {case.codec_name}, got {codec.getName()}")
    codec_cache[case.codec_name] = codec
    return codec


def run_case(case, codec_class, codec_cache, jarray):
    from java.nio.file import Paths
    from org.apache.lucene.document import Document, Field, KnnFloatVectorField, StringField
    from org.apache.lucene.index import (
        DirectoryReader,
        IndexWriter,
        IndexWriterConfig,
        Term,
        VectorSimilarityFunction,
    )
    from org.apache.lucene.search import IndexSearcher
    from org.apache.lucene.store import FSDirectory

    available_codecs = codec_class.availableCodecs()
    if not available_codecs.contains(case.codec_name):
        raise AssertionError(
            f"{case.codec_name} was not advertised by Lucene SPI. "
            f"Available codecs: {available_codecs}"
        )
    codec = codec_for_case(case, codec_class, codec_cache)

    missing_ids = missing_doc_ids(case)
    deleted_ids = deleted_doc_ids(case, missing_ids)
    active_vector_ids = [
        doc_id
        for doc_id in range(case.row_count)
        if doc_id not in missing_ids and doc_id not in deleted_ids
    ]
    if not active_vector_ids:
        raise AssertionError(f"{case.name}: no active vectors available for search")
    query_ids = choose_query_ids(case.row_count, active_vector_ids)
    inactive_doc_names = {f"doc-{doc_id}" for doc_id in missing_ids | deleted_ids}
    writer_path = "unknown"

    with ExitStack() as stack:
        stack.enter_context(hnsw_cpu_fallback(case))
        index_path = stack.enter_context(
            tempfile.TemporaryDirectory(prefix=f"cuvs-lucene-pylucene-{case.name}-")
        )
        directory = FSDirectory.open(Paths.get(index_path))
        config = IndexWriterConfig()
        config.setCodec(codec)
        config.setUseCompoundFile(False)
        if case.disable_background_merges:
            from org.apache.lucene.index import NoMergePolicy

            config.setMergePolicy(NoMergePolicy.INSTANCE)

        writer = IndexWriter(directory, config)
        try:
            segment_ends = segment_end_doc_ids(case)
            for doc_id in range(case.row_count):
                doc = Document()
                doc.add(StringField(ID_FIELD, f"doc-{doc_id}", Field.Store.YES))
                doc.add(StringField(GROUP_FIELD, f"group-{doc_id % 3}", Field.Store.YES))
                if doc_id not in missing_ids:
                    doc.add(
                        KnnFloatVectorField(
                            VECTOR_FIELD,
                            fvec(jarray, vector_for(doc_id, case.dims)),
                            VectorSimilarityFunction.EUCLIDEAN,
                        )
                    )
                writer.addDocument(doc)
                if doc_id in segment_ends:
                    writer.commit()

            writer.commit()
            for doc_id in deleted_ids:
                writer.deleteDocuments(Term(ID_FIELD, f"doc-{doc_id}"))
            if deleted_ids:
                writer.commit()
            if case.force_merge_target:
                writer.forceMerge(case.force_merge_target)
                writer.commit()
        finally:
            writer.close()

        assert_expected_index_files(index_path, case.expected_suffixes)

        reader = DirectoryReader.open(directory)
        try:
            expected_live_docs = case.row_count - len(deleted_ids)
            if reader.numDocs() != expected_live_docs:
                raise AssertionError(
                    f"{case.name}: expected {expected_live_docs} live docs, got {reader.numDocs()}"
                )
            expected_vector_count = (
                len(active_vector_ids)
                if case.force_merge_target
                else case.row_count - len(missing_ids)
            )
            assert_vector_metadata(reader, VECTOR_FIELD, expected_vector_count, case.dims)
            assert_segment_topology(reader, case)
            searcher = IndexSearcher(reader)
            assert_search_results(searcher, jarray, case, query_ids, inactive_doc_names)
            assert_filtered_search(searcher, jarray, case, query_ids[0])
            telemetry = writer_telemetry(codec)
            writer_path = observed_writer_path(case, telemetry)
            assert_hnsw_telemetry(case, telemetry)
        finally:
            reader.close()
            directory.close()

    algorithm_label = build_search_label(case, writer_path)
    print(
        "PASS: "
        f"{case.name} loaded {case.codec_name}, "
        f"indexed {case.row_count} docs x {case.dims} dims, "
        f"segments={effective_segment_count(case)}"
        f"{'->' + str(case.force_merge_target) if case.force_merge_target else ''}, "
        f"searched topK={case.top_k}, "
        f"{algorithm_label}, "
        f"path={writer_path}"
    )


def initialize_pylucene_context():
    cuvs_lucene_jar = find_cuvs_lucene_jar()
    cuvs_java_jar = find_cuvs_java_jar()
    init_vm(cuvs_java_jar, cuvs_lucene_jar)

    from lucene import JArray
    from java.lang import Class

    # Forces an early, clear failure if cuvs-java was flattened or omitted.
    Class.forName("com.nvidia.cuvs.spi.JDKProvider")

    quiet_expected_provider_fallback_logs()
    print_threadlocal_provider_diagnostics()

    from org.apache.lucene.codecs import Codec

    verify_codecs_advertised(Codec)

    return PyLuceneContext(
        cuvs_lucene_jar=cuvs_lucene_jar,
        cuvs_java_jar=cuvs_java_jar,
        codec_class=Codec,
        jarray=JArray,
        codec_cache={},
    )


def main():
    import pytest

    test_path = Path(__file__).with_name("test_pylucene_smoke.py")
    return pytest.main(["-q", "-s", str(test_path)])


if __name__ == "__main__":
    raise SystemExit(main())
