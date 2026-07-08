# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import tempfile
from pathlib import Path

import lucene


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CODEC = "Lucene101AcceleratedHNSWCodec"
ID_FIELD = "id"
VECTOR_FIELD = "vector"


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


def main():
    cuvs_lucene_jar = find_cuvs_lucene_jar()
    cuvs_java_jar = find_cuvs_java_jar()
    init_vm(cuvs_java_jar, cuvs_lucene_jar)

    from lucene import JArray
    from java.lang import Class
    from java.nio.file import Paths
    from org.apache.lucene.codecs import Codec
    from org.apache.lucene.document import Document, Field, KnnFloatVectorField, StringField
    from org.apache.lucene.index import (
        DirectoryReader,
        IndexWriter,
        IndexWriterConfig,
        VectorSimilarityFunction,
    )
    from org.apache.lucene.search import IndexSearcher, KnnFloatVectorQuery
    from org.apache.lucene.store import FSDirectory

    # Forces an early, clear failure if cuvs-java was flattened or omitted.
    Class.forName("com.nvidia.cuvs.spi.JDKProvider")

    codec_name = os.environ.get("CUVS_LUCENE_PYLUCENE_CODEC", DEFAULT_CODEC)
    row_count = int(os.environ.get("CUVS_LUCENE_PYLUCENE_ROWS", "2"))
    dims = int(os.environ.get("CUVS_LUCENE_PYLUCENE_DIMS", "3"))
    top_k = int(os.environ.get("CUVS_LUCENE_PYLUCENE_TOPK", "2"))
    expect_cuvs_files = os.environ.get("CUVS_LUCENE_EXPECT_CUVS_FILES") == "1"

    available_codecs = Codec.availableCodecs()
    if not available_codecs.contains(codec_name):
        raise AssertionError(
            f"{codec_name} was not advertised by Lucene SPI. "
            f"Available codecs: {available_codecs}"
        )

    codec = Codec.forName(codec_name)
    if codec.getName() != codec_name:
        raise AssertionError(f"Expected codec {codec_name}, got {codec.getName()}")

    with tempfile.TemporaryDirectory(prefix="cuvs-lucene-pylucene-") as index_path:
        directory = FSDirectory.open(Paths.get(index_path))
        config = IndexWriterConfig()
        config.setCodec(codec)
        config.setUseCompoundFile(False)

        writer = IndexWriter(directory, config)
        try:
            for doc_id in range(row_count):
                doc = Document()
                doc.add(StringField(ID_FIELD, f"doc-{doc_id}", Field.Store.YES))
                doc.add(
                    KnnFloatVectorField(
                        VECTOR_FIELD,
                        fvec(JArray, vector_for(doc_id, dims)),
                        VectorSimilarityFunction.EUCLIDEAN,
                    )
                )
                writer.addDocument(doc)
            writer.commit()
        finally:
            writer.close()

        if expect_cuvs_files:
            index_files = sorted(path.name for path in Path(index_path).iterdir())
            if not any(name.endswith(".vcag") for name in index_files):
                raise AssertionError(f"No cuVS .vcag file found: {index_files}")
            if not any(name.endswith(".vemc") for name in index_files):
                raise AssertionError(f"No cuVS .vemc file found: {index_files}")

        reader = DirectoryReader.open(directory)
        try:
            searcher = IndexSearcher(reader)
            stored_fields = searcher.storedFields()
            query_ids = sorted({0, row_count // 2, row_count - 1})
            for query_id in query_ids:
                query = KnnFloatVectorQuery(
                    VECTOR_FIELD, fvec(JArray, vector_for(query_id, dims)), top_k
                )
                hits = searcher.search(query, top_k).scoreDocs
                ids = [stored_fields.document(hit.doc).get(ID_FIELD) for hit in hits]
                expected = f"doc-{query_id}"
                if expected not in ids:
                    raise AssertionError(
                        f"Expected {expected} in top {top_k}, got {ids}"
                    )
        finally:
            reader.close()
            directory.close()

    print(
        "PyLucene smoke test passed: "
        f"loaded {codec_name} from {cuvs_lucene_jar.name}, "
        f"used {cuvs_java_jar.name}, and searched {row_count} docs"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"PyLucene smoke test failed: {exc}", file=sys.stderr)
        raise
