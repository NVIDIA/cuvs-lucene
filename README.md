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

Build the PyLucene sidecar jar:

```sh
mvn clean package -DskipTests
```

Then start PyLucene with the base `cuvs-java` jar, the generated PyLucene
sidecar jar, and PyLucene's own Lucene classpath:

```python
import os
from pathlib import Path

import lucene

cuvs_java_jar = Path(os.environ["CUVS_LUCENE_CUVS_JAVA_JAR"])
cuvs_lucene_jar = next(
    Path("target").glob("cuvs-lucene-*-jar-with-pylucene-dependencies.jar")
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

Use the returned `codec` with `IndexWriterConfig.setCodec(codec)`. The
`jar-with-pylucene-dependencies` artifact includes only `cuvs-lucene` classes and
service descriptors. PyLucene must provide Lucene classes, and the base
multi-release `cuvs-java` jar must be present separately on the JVM classpath. Do
not use a native classifier `cuvs-java` jar here unless you also want to rely on
its embedded native libraries; the base jar uses native libraries from
`LD_LIBRARY_PATH`/`java.library.path`.

To run an end-to-end smoke test against a local PyLucene environment:

```sh
./ci/test_pylucene_smoke.sh
```

To run the same smoke through the GPU search codec with 2,000 documents:

```sh
./ci/test_pylucene_smoke.sh --gpu-e2e
```

### Running Tests

```sh
export LD_LIBRARY_PATH={ PATH TO YOUR LOCAL libcuvs_c.so }:$LD_LIBRARY_PATH && mvn clean test
```

## Contributing

> [!NOTE]
> The code style format is automatically enforced (including the missing license header, if any) using the [Spotless maven plugin](https://github.com/diffplug/spotless/tree/main/plugin-maven). This currently happens in the maven's `validate` stage.
