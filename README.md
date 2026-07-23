# cuVS Lucene

This is a project for using [cuVS](https://github.com/rapidsai/cuvs), NVIDIA's GPU accelerated vector search library, with [Apache Lucene](https://github.com/apache/lucene).

## Contents

1. [What is cuvs-lucene?](#what-is-cuvs-lucene)
2. [Installing cuvs-lucene](#installing-cuvs-lucene)
3. [Getting Started](#getting-started)
4. [Contributing](#contributing)
5. [References](#references)

## What is cuvs-lucene?

`cuvs-lucene` provides a pluggable [KnnVectorsFormat](https://lucene.apache.org/core/10_2_0/core/org/apache/lucene/codecs/KnnVectorsFormat.html) that uses cuVS to offload vector index build — and optionally search — to NVIDIA GPUs. Because it plugs in through a standard Lucene codec, existing Lucene applications can take advantage of GPU acceleration with minimal code changes and gracefully fall back to the default CPU codec when no GPU is present.

Four codecs are currently provided:

- `Lucene101AcceleratedHNSWCodec` — GPU-accelerated HNSW build with CPU HNSW search. The on-disk format is standard Lucene HNSW, so indexes built on the GPU can be read by any stock Lucene 10.x reader.
  - `LuceneAcceleratedHNSWScalarQuantizedCodec` — scalar-quantized vectors for a smaller index footprint.
  - `LuceneAcceleratedHNSWBinaryQuantizedCodec` — binary-quantized vectors for an even smaller index footprint.
- `CuVS2510GPUSearchCodec` — GPU-accelerated HNSW build and GPU search

## Installing cuvs-lucene

### Prerequisites

- [CUDA 12.0+](https://developer.nvidia.com/cuda-toolkit-archive)
- [JDK 22](https://jdk.java.net/archive/)
- [Maven 3.9.6+](https://maven.apache.org/download.cgi)
- A compatible cuVS installation (26.04 - 26.06). For Maven usage, install the cuVS tarball and add it to your system library load path. See the cuVS [tarball install instructions](https://docs.rapids.ai/api/cuvs/stable/build/#download-extract).

### Maven

To pull `cuvs-lucene` into a Maven project, add the following dependency to your `pom.xml`:

```xml
<dependency>
  <groupId>com.nvidia.cuvs.lucene</groupId>
  <artifactId>cuvs-lucene</artifactId>
  <version>26.06.0</version>
</dependency>
```

### Building from source

```sh
git clone https://github.com/rapidsai/cuvs-lucene.git
cd cuvs-lucene
mvn clean compile package
```

The resulting artifacts are written to `target/`. To run the tests, first install cuVS and add it to your system library load path, as described in the cuVS [tarball install instructions](https://docs.rapids.ai/api/cuvs/stable/build/#download-extract), then run:

```sh
mvn clean test
```

## Getting Started

The example below plugs the GPU-accelerated HNSW codec into a standard Lucene `IndexWriter`. Once the codec is set on the `IndexWriterConfig`, indexing proceeds exactly as it would with the default Lucene codec, and search uses the stock `KnnFloatVectorQuery`.

Before running it, make sure cuVS is installed and available on your system library load path. The cuVS [tarball install instructions](https://docs.rapids.ai/api/cuvs/stable/build/#download-extract) show how to set this up.

In a Maven project that includes the `cuvs-lucene` dependency shown above, create `src/main/java/com/nvidia/cuvs/lucene/examples/HelloCuvsLucene.java`:

```java
package com.nvidia.cuvs.lucene.examples;

import static org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN;

import com.nvidia.cuvs.lucene.AcceleratedHNSWParams;
import com.nvidia.cuvs.lucene.Lucene101AcceleratedHNSWCodec;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

public class HelloCuvsLucene {
  public static void main(String[] args) throws Exception {
    AcceleratedHNSWParams params = new AcceleratedHNSWParams.Builder().build();
    Codec codec = new Lucene101AcceleratedHNSWCodec(params);
    IndexWriterConfig config = new IndexWriterConfig().setCodec(codec);

    Path indexPath = Paths.get("index");
    float[] embedding = new float[] {0.1f, 0.2f, 0.3f, 0.4f};

    try (Directory dir = FSDirectory.open(indexPath);
        IndexWriter writer = new IndexWriter(dir, config)) {
      Document doc = new Document();
      doc.add(new KnnFloatVectorField("vector_field", embedding, EUCLIDEAN));
      writer.addDocument(doc);
    }

    System.out.println("Hello cuVS Lucene ran successfully.");
  }
}
```

The artifacts would be built and available in the target / folder.

### Using with PyLucene

PyLucene embeds a JVM and starts it with the classpath passed to `lucene.initVM(...)`.
Because PyLucene's generated Python module only exposes the Java classes it was built
to wrap, use Lucene's service provider lookup to load `cuvs-lucene` codecs from
Python instead of importing `com.nvidia.cuvs.lucene` classes directly.

Build the standard cuvs-lucene jar:

```sh
mvn clean package -DskipTests
```

Then start PyLucene with the base `cuvs-java` jar, the standard `cuvs-lucene`
jar, and PyLucene's own Lucene classpath:

```python
import os
from pathlib import Path

import lucene

cuvs_java_jar = Path(os.environ["CUVS_LUCENE_CUVS_JAVA_JAR"])
cuvs_lucene_jar = next(
    jar
    for jar in Path("target").glob("cuvs-lucene-*.jar")
    if "-jar-with-" not in jar.name
    and not jar.name.endswith(("-sources.jar", "-javadoc.jar"))
)
lucene.initVM(
    classpath=os.pathsep.join(
        [str(cuvs_java_jar), str(cuvs_lucene_jar), lucene.CLASSPATH]
    ),
    vmargs=[
        "--enable-native-access=ALL-UNNAMED",
        "--add-modules=jdk.incubator.vector",
    ],
)

from org.apache.lucene.codecs import Codec

codec = Codec.forName("Lucene101AcceleratedHNSWCodec")
```

Use the returned `codec` with `IndexWriterConfig.setCodec(codec)`. The standard
artifact includes `cuvs-lucene` classes and service descriptors.
PyLucene must provide Lucene classes, and the base multi-release `cuvs-java` jar
must be present separately on the JVM classpath. Do not use a native classifier
`cuvs-java` jar here unless you also want to rely on its embedded native
libraries; the base jar uses native libraries from
`LD_LIBRARY_PATH`/`java.library.path`.

To run the PyLucene pytest smoke suite against a local PyLucene environment:

```sh
./test_pylucene.sh
```

The script builds and validates the jar before invoking pytest. To invoke pytest
directly against existing artifacts instead:

```sh
CUVS_LUCENE_JAR=/path/to/cuvs-lucene.jar \
CUVS_LUCENE_CUVS_JAVA_JAR=/path/to/cuvs-java.jar \
python3 -m pytest -q -s examples/Python/test_pylucene_smoke.py
```

To run an expanded GPU end-to-end pytest suite through CPU HNSW,
CAGRA-to-HNSW, and CAGRA search paths:

```sh
./test_pylucene.sh --gpu-e2e
```

The expanded suite runs the `gpu-basic`, `gpu-segments`, `cpu-hnsw`, and
`cagra-hnsw` case groups. The basic cases cover `hnsw`, `cagra`, `hnsw-single`,
and `cagra-single`. The segment cases cover 1-segment indexes, 10-segment
indexes, 10 segments force-merged to 1, and 100 segments force-merged to 10 for
both HNSW and CAGRA. The CPU HNSW cases force the accelerated HNSW codec through
its Lucene CPU fallback path in the same run, including 10 segments force-merged
to 1 and 100 segments force-merged to 10. The CAGRA-to-HNSW cases explicitly
cover one-layer and three-layer HNSW graphs built from CAGRA with NN_DESCENT,
`graphDegree=32`, and `intermediateGraphDegree=64`. The base matrix uses 2,000
documents and 32 dimensions; high-segment cases use at least 257 rows per
segment to avoid expected cuVS graph-degree clamps on tiny per-segment datasets.
The suite checks Lucene SPI discovery, jar packaging, index file suffixes
(`.vex`/`.vem` for HNSW and `.vcag`/`.vemc` for CAGRA), indexed vector metadata,
unfiltered KNN, filtered KNN, missing-vector documents, deletions, and force
merge behavior. To run a subset or resize the test:

```sh
./test_pylucene.sh --gpu-e2e --cases=gpu-segments --rows=5000 --dims=64 --topk=20
```

### Running Tests

```sh
mvn -q compile org.codehaus.mojo:exec-maven-plugin:3.5.1:java \
  -Dexec.mainClass=com.nvidia.cuvs.lucene.examples.HelloCuvsLucene
```

For more examples, including one that indexes and searches entirely on the GPU using `CuVS2510GPUSearchCodec`, please refer to the [`examples/`](examples) directory.

## Contributing

If you are interested in contributing to cuvs-lucene, please read our [Contributing guide](CONTRIBUTING.md).

> [!NOTE]
> The code style format is automatically enforced (including the missing license header, if any) using the [Spotless maven plugin](https://github.com/diffplug/spotless/tree/main/plugin-maven). This currently happens in the maven's `validate` stage.

## References

- [Bring Massive-Scale Vector Search to the GPU with Apache Lucene](https://www.nvidia.com/en-us/on-demand/session/gtc25-S71286/) — NVIDIA GTC 2025 session video
- [cuVS and Lucene: GPU-based Vector Search](https://www.youtube.com/watch?v=qiW7iIDFJC0) — Berlin Buzzwords 2024 session video
- [Exploring GPU-accelerated vector search in Elasticsearch with NVIDIA](https://www.elastic.co/search-labs/blog/gpu-accelerated-vector-search-elasticsearch-nvidia) — Elasticsearch Blog
- [Apache Lucene Accelerated with the NVIDIA cuVS 25.06 Release](https://searchscale.com/blog/apache-lucene-accelerated-with-nvidia-cuvs-25.06-release/) — SearchScale Blog
