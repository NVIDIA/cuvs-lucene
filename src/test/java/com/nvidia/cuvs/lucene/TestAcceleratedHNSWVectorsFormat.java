/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.TestDataProvider.ID_FIELD;
import static com.nvidia.cuvs.lucene.TestDataProvider.VECTOR_FIELD1;
import static com.nvidia.cuvs.lucene.TestDataProvider.VECTOR_FIELD2;
import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.isSupported;
import static org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN;

import java.util.List;
import java.util.Random;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.VectorEncoding;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.index.BaseKnnVectorsFormatTestCase;
import org.apache.lucene.tests.util.LuceneTestCase.SuppressSysoutChecks;
import org.apache.lucene.tests.util.TestUtil;
import org.junit.BeforeClass;
import org.junit.Ignore;

@SuppressSysoutChecks(bugUrl = "")
public class TestAcceleratedHNSWVectorsFormat extends BaseKnnVectorsFormatTestCase {

  private static TestDataProvider dataProvider;
  private static Random random;

  @BeforeClass
  public static void beforeClass() {
    assumeTrue("cuVS is not supported", isSupported());
    random = random();
    dataProvider = new TestDataProvider(random);
  }

  @Override
  protected Codec getCodec() {
    return TestUtil.alwaysKnnVectorsFormat(new Lucene99AcceleratedHNSWVectorsFormat());
  }

  public void testMergeTwoSegsWithASingleDocPerSeg() throws Exception {
    final int numDocs = 2;
    float[][] vectors = dataProvider.getVectors(numDocs);

    try (Directory dir = newDirectory();
        IndexWriter w = new IndexWriter(dir, newIndexWriterConfig())) {

      for (int i = 0; i < numDocs; i++) {
        Document doc = new Document();
        doc.add(new StringField(ID_FIELD, String.valueOf(i), Field.Store.YES));
        doc.add(new KnnFloatVectorField(VECTOR_FIELD1, vectors[i], EUCLIDEAN));
        w.addDocument(doc);
        w.commit();
      }

      // sanity - verify one doc per leaf
      try (DirectoryReader reader = DirectoryReader.open(w)) {
        List<LeafReaderContext> subReaders = reader.leaves();
        assertEquals(numDocs, subReaders.size());
        for (int i = 0; i < numDocs; i++) {
          assertEquals(1, subReaders.get(i).reader().getFloatVectorValues(VECTOR_FIELD1).size());
        }
      }

      // now merge to a single segment
      w.forceMerge(1);

      // verify merged content
      try (DirectoryReader reader = DirectoryReader.open(w)) {
        LeafReader r = getOnlyLeafReader(reader);
        FloatVectorValues values = r.getFloatVectorValues(VECTOR_FIELD1);
        assertNotNull(values);
        assertEquals(numDocs, values.size());
        for (int i = 0; i < numDocs; i++) {
          assertArrayEquals(vectors[i], values.vectorValue(i), 0.0f);
        }
      }
    }
  }

  // Basic test for multiple vectors fields per document
  public void testTwoVectorFieldsPerDoc() throws Exception {

    final int numDocs = 2;
    float[][] vectors1 = dataProvider.getVectors(numDocs);
    float[][] vectors2 = dataProvider.getVectors(numDocs);

    try (Directory dir = newDirectory();
        IndexWriter w = new IndexWriter(dir, newIndexWriterConfig())) {

      for (int i = 0; i < numDocs; i++) {
        Document doc = new Document();
        doc.add(new StringField(ID_FIELD, String.valueOf(i), Field.Store.YES));
        doc.add(new KnnFloatVectorField(VECTOR_FIELD1, vectors1[i], EUCLIDEAN));
        doc.add(new KnnFloatVectorField(VECTOR_FIELD2, vectors2[i], EUCLIDEAN));
        w.addDocument(doc);
      }

      w.forceMerge(1);

      try (DirectoryReader reader = DirectoryReader.open(w)) {
        LeafReader r = getOnlyLeafReader(reader);

        for (int i = 0; i < numDocs; i++) {
          FloatVectorValues values = r.getFloatVectorValues(VECTOR_FIELD1);
          assertNotNull(values);
          assertEquals(2, values.size());
          assertArrayEquals(vectors1[i], values.vectorValue(i), 0.0f);
        }

        for (int i = 0; i < numDocs; i++) {
          FloatVectorValues values = r.getFloatVectorValues(VECTOR_FIELD2);
          assertNotNull(values);
          assertEquals(2, values.size());
          assertArrayEquals(vectors2[i], values.vectorValue(i), 0.0f);
        }

        // Check boundary condition - search with a 0 topK
        float[] target = dataProvider.getVectors(1)[0];
        var topDocs = r.searchNearestVectors(VECTOR_FIELD1, target, 0, null, 10);
        assertEquals(0, topDocs.scoreDocs.length);
        assertEquals(0, topDocs.totalHits.value());
      }
    }
  }

  @Override
  // Overriding this method from superclass for the tests to only use float vector encoding
  protected VectorEncoding randomVectorEncoding() {
    return VectorEncoding.FLOAT32;
  }

  @Ignore
  @Override
  // Ignoring this test from superclass as we do not support byte vectors
  public void testByteVectorScorerIteration() {}

  @Ignore
  @Override
  // Ignoring this test from superclass as we do not support byte vectors
  public void testEmptyByteVectorData() {}

  @Ignore
  @Override
  // Ignoring this test from superclass as we do not support byte vectors
  public void testMergingWithDifferentByteKnnFields() {}

  @Ignore
  @Override
  // Ignoring this test from superclass as we do not support byte vectors
  public void testMismatchedFields() {}

  @Ignore
  @Override
  // Ignoring this test from superclass as we do not support byte vectors
  public void testRandomBytes() {}

  @Ignore
  @Override
  // Ignoring this test from superclass as we do not support byte vectors
  public void testSortedIndexBytes() {}
}
