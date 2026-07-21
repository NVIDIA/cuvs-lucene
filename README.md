# Lucene cuVS

This is a project for using [cuVS](https://github.com/rapidsai/cuvs), NVIDIA's GPU accelerated vector search library, with [Apache Lucene](https://github.com/apache/lucene).

## Overview

This library provides a new [KnnVectorFormat](https://lucene.apache.org/core/10_3_1/core/org/apache/lucene/codecs/KnnVectorsFormat.html) which can be plugged into a Lucene codec.

## Building

### Prerequisites

- [CUDA 12.0+](https://developer.nvidia.com/cuda-toolkit-archive),
- [Maven 3.9.6+](https://maven.apache.org/download.cgi),
- [JDK 22](https://jdk.java.net/archive/)

```sh
mvn clean compile package
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
CUVS_LUCENE_JAR=target/cuvs-lucene-26.08.0.jar \
CUVS_LUCENE_CUVS_JAVA_JAR=/path/to/cuvs-java-26.08.0.jar \
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
export LD_LIBRARY_PATH={ PATH TO YOUR LOCAL libcuvs_c.so }:$LD_LIBRARY_PATH && mvn clean test
```

## Contributing

> [!NOTE]
> The code style format is automatically enforced (including the missing license header, if any) using the [Spotless maven plugin](https://github.com/diffplug/spotless/tree/main/plugin-maven). This currently happens in the maven's `validate` stage.
