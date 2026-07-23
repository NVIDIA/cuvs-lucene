/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import com.nvidia.cuvs.CagraIndexParams.CagraGraphBuildAlgo;

/**
 * Accelerated HNSW codec that builds an intermediate CAGRA graph and writes only the base HNSW layer.
 */
public class Lucene101AcceleratedHNSWBaseLayerCodec extends Lucene101AcceleratedHNSWCodec {

  private static final String NAME = "Lucene101AcceleratedHNSWBaseLayerCodec";
  private static final int CAGRA_GRAPH_DEGREE = 32;
  private static final int CAGRA_INTERMEDIATE_GRAPH_DEGREE = 64;

  /** Default constructor used by Lucene SPI. */
  public Lucene101AcceleratedHNSWBaseLayerCodec() throws Exception {
    super(
        NAME,
        LuceneProvider.getDefaultDelegateCodec(),
        new AcceleratedHNSWParams.Builder()
            .withStrategy(AcceleratedHNSWParams.Strategy.CUSTOM)
            .withCagraGraphBuildAlgo(CagraGraphBuildAlgo.NN_DESCENT)
            .withGraphDegree(CAGRA_GRAPH_DEGREE)
            .withIntermediateGraphDegree(CAGRA_INTERMEDIATE_GRAPH_DEGREE)
            .withHNSWLayer(1)
            .build());
  }
}
