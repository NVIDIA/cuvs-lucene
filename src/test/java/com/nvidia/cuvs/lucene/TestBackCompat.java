/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertTrue;

import java.util.Set;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.codecs.hnsw.FlatVectorsFormat;
import org.junit.Test;

/**
 * Tests the backward compatibility mechanism.
 *
 * @since 25.12
 */
public class TestBackCompat {

  @Test
  public void testFallback() throws Exception {
    // Lucene99Codec exists in the org.apache.lucene.backward_codecs.lucene99
    Codec c = LuceneProvider.getCodec("99");
    assertEquals(c.getName(), "Lucene99");
  }

  @Test(expected = ClassNotFoundException.class)
  public void testNonexistentCodec() throws Exception {
    LuceneProvider.getCodec("0");
  }

  @Test
  public void testExistingComponents() throws Exception {
    LuceneProvider provider = LuceneProvider.getInstance("99");
    assertTrue(provider.getLuceneFlatVectorsFormatInstance(null) instanceof FlatVectorsFormat);
    assertEquals(provider.getStaticIntParam("VERSION_CURRENT"), 0);
    assertNotEquals(provider.getSimilarityFunctions().size(), 0);
  }

  @Test
  public void testProviderCachesSupportedVersion() throws Exception {
    LuceneProvider provider99 = LuceneProvider.getInstance("99");
    assertSame(provider99, LuceneProvider.getInstance("99"));
  }

  @Test(expected = ClassNotFoundException.class)
  public void testProviderDoesNotPretendLucene102IsACompleteProviderVersion() throws Exception {
    LuceneProvider.getInstance("102");
  }

  @Test
  public void testDefaultDelegateCodec() {
    Codec delegate = LuceneProvider.getDefaultDelegateCodec();
    assertNotNull(delegate);
    assertTrue(Set.of("Lucene101", "Lucene99").contains(delegate.getName()));
    assertTrue(delegate.getClass().getName().startsWith("org.apache.lucene."));
  }

  @Test
  public void testServiceLoadedCodecsCanBeInstantiated() {
    String[] codecNames = {
      "Lucene101AcceleratedHNSWCodec",
      "Lucene101AcceleratedHNSWBaseLayerCodec",
      "Lucene101AcceleratedHNSWMultiLayerCodec",
      "CuVS2510GPUSearchCodec",
      "Lucene101AcceleratedHNSWBinaryQuantizedCodec",
      "Lucene101AcceleratedHNSWScalarQuantizedCodec"
    };
    for (String codecName : codecNames) {
      assertTrue(Codec.availableCodecs().contains(codecName));
      assertEquals(codecName, Codec.forName(codecName).getName());
    }
  }
}
