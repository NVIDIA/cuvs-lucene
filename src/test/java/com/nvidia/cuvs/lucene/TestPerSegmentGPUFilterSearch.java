/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.TestUtils.generateDataset;
import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.isSupported;
import static org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN;

import java.util.HashSet;
import java.util.Set;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.store.ByteBuffersDirectory;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.LuceneTestCase.SuppressSysoutChecks;
import org.apache.lucene.tests.util.TestUtil;
import org.junit.Test;

/**
 * Exercises the per-segment (single-index) GPU search fallback with a selective prefilter.
 *
 * <p>{@code TestUtil.alwaysKnnVectorsFormat} wraps the segment reader, so {@code
 * GPUKnnFloatVectorQuery} cannot unwrap it to a {@link CuVS2510GPUVectorsReader} and falls back to
 * per-segment search ({@link CuVS2510GPUVectorsReader#search}) rather than the multi-partition path.
 * A highly selective filter rejects the trailing ordinals of the segment — the regime where the
 * per-segment prefilter previously mis-sized itself and default-accepted those ordinals, returning
 * filtered-out vectors. Every returned hit must satisfy its filter.
 */
@SuppressSysoutChecks(bugUrl = "")
public class TestPerSegmentGPUFilterSearch extends LuceneTestCase {

  private static final Codec codec =
      TestUtil.alwaysKnnVectorsFormat(new CuVS2510GPUVectorsFormat());
  private static final String VECTOR_FIELD = "vectors";
  private static final String CATEGORY_FIELD = "cat";
  private static final int NUM_CATEGORIES = 24;

  @Test
  public void testSelectiveFilterDoesNotLeakTrailingOrdinals() throws Exception {
    assumeTrue("cuVS not supported", isSupported());

    // A segment size that is not a multiple of 32 exercises the last partial uint32 word of the
    // prefilter bitset, which is where the trailing ordinals used to leak.
    final int datasetSize = 500;
    final int dimensions = 128;
    final int topK = 10;

    try (Directory directory = newDirectory(new ByteBuffersDirectory())) {
      float[][] dataset = generateDataset(random(), datasetSize, dimensions);
      IndexWriterConfig config = new IndexWriterConfig().setCodec(codec);
      try (IndexWriter writer = new IndexWriter(directory, config)) {
        for (int i = 0; i < datasetSize; i++) {
          Document doc = new Document();
          doc.add(new StringField(CATEGORY_FIELD, "c" + (i % NUM_CATEGORIES), Field.Store.NO));
          doc.add(new KnnFloatVectorField(VECTOR_FIELD, dataset[i], EUCLIDEAN));
          writer.addDocument(doc);
        }
        writer.commit();
      }

      try (DirectoryReader reader = DirectoryReader.open(directory)) {
        IndexSearcher searcher = new IndexSearcher(reader);
        float[][] queries = generateDataset(random(), 32, dimensions);

        // Each category accepts ~1/24 of the docs, so the trailing ordinals are rejected.
        for (int c = 0; c < NUM_CATEGORIES; c++) {
          Query filter = new TermQuery(new Term(CATEGORY_FIELD, "c" + c));
          Set<Integer> accepted = new HashSet<>();
          for (ScoreDoc sd : searcher.search(filter, datasetSize).scoreDocs) {
            accepted.add(sd.doc);
          }
          for (float[] q : queries) {
            GPUKnnFloatVectorQuery query =
                new GPUKnnFloatVectorQuery(VECTOR_FIELD, q, topK, filter, topK, 1);
            for (ScoreDoc hit : searcher.search(query, topK).scoreDocs) {
              assertTrue(
                  "per-segment search returned doc " + hit.doc + " not accepted by filter c" + c,
                  accepted.contains(hit.doc));
            }
          }
        }
      }
    }
  }
}
