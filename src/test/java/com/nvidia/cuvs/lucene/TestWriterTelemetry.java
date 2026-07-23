/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;

import org.junit.Test;

public class TestWriterTelemetry {

  private static final String FORCE_CPU_HNSW_FALLBACK_PROPERTY = "cuvs.lucene.forceCpuHnswFallback";

  @Test
  public void testCagraTelemetryIsComputedOnDemand() {
    assertEquals("writerPath=gpu-cagra", WriterTelemetry.forCagra());
    assertEquals(
        "CuVS2510GPUVectorsFormat(writerPath=gpu-cagra)",
        new CuVS2510GPUVectorsFormat().toString());
    assertNull(System.getProperty("cuvs.lucene.lastCagraWriterPath"));
  }

  @Test
  public void testForcedCpuHnswTelemetryIsComputedOnDemand() {
    String previousValue = System.getProperty(FORCE_CPU_HNSW_FALLBACK_PROPERTY);
    System.setProperty(FORCE_CPU_HNSW_FALLBACK_PROPERTY, "true");
    try {
      AcceleratedHNSWParams params =
          new AcceleratedHNSWParams.Builder()
              .withHNSWLayer(3)
              .withGraphDegree(32)
              .withIntermediateGraphDegree(64)
              .build();

      assertEquals(
          "writerPath=cpu-hnsw-fallback;hnswLayers=3;"
              + "cagraGraphBuildAlgo=NN_DESCENT;cagraGraphDegree=32;"
              + "cagraIntermediateGraphDegree=64",
          WriterTelemetry.forHnsw(params));
      assertEquals(
          "Lucene99AcceleratedHNSWVectorsFormat("
              + "writerPath=cpu-hnsw-fallback;hnswLayers=3;"
              + "cagraGraphBuildAlgo=NN_DESCENT;cagraGraphDegree=32;"
              + "cagraIntermediateGraphDegree=64)",
          new Lucene99AcceleratedHNSWVectorsFormat(params).toString());
      assertNull(System.getProperty("cuvs.lucene.lastHnswWriterPath"));
      assertNull(System.getProperty("cuvs.lucene.lastHnswLayers"));
    } finally {
      if (previousValue == null) {
        System.clearProperty(FORCE_CPU_HNSW_FALLBACK_PROPERTY);
      } else {
        System.setProperty(FORCE_CPU_HNSW_FALLBACK_PROPERTY, previousValue);
      }
    }
  }
}
