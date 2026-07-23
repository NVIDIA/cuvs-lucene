/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.TestDataProvider.ID_FIELD;
import static com.nvidia.cuvs.lucene.TestDataProvider.TEXT_FIELD;
import static com.nvidia.cuvs.lucene.TestDataProvider.VECTOR_FIELD1;
import static com.nvidia.cuvs.lucene.TestUtils.createWriter;
import static com.nvidia.cuvs.lucene.TestUtils.generateExpectedTopK;
import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.isSupported;
import static org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN;

import java.io.IOException;
import java.util.List;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.index.RandomIndexWriter;
import org.apache.lucene.tests.util.English;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.LuceneTestCase.SuppressSysoutChecks;
import org.apache.lucene.tests.util.TestUtil;
import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;

@SuppressSysoutChecks(bugUrl = "")
public class TestCuVSGaps extends LuceneTestCase {

  private static final Logger log = Logger.getLogger(TestCuVSGaps.class.getName());
  private static Codec codec;
  private static IndexSearcher searcher;
  private static IndexReader reader;
  private static Directory directory;
  private static Random random;
  private static TestDataProvider dataProvider;

  @BeforeClass
  public static void beforeClass() throws Exception {
    assumeTrue("cuVS not supported", isSupported());
    directory = newDirectory();
    random = random();
    dataProvider = new TestDataProvider(random);

    codec = TestUtil.alwaysKnnVectorsFormat(new CuVS2510GPUVectorsFormat());
    RandomIndexWriter writer = createWriter(random, directory, codec);
    int datasetSize = dataProvider.getDatasetSize();
    float[][] dataset = dataProvider.getDataset1();

    // Create documents where only even-numbered documents have vectors
    for (int i = 0; i < datasetSize; i++) {
      Document doc = new Document();
      doc.add(new StringField(ID_FIELD, String.valueOf(i), Field.Store.YES));
      doc.add(newTextField(TEXT_FIELD, English.intToEnglish(i), Field.Store.YES));

      // Only add vectors to even-numbered documents
      if (i % 2 == 0) {
        doc.add(new KnnFloatVectorField(VECTOR_FIELD1, dataset[i], EUCLIDEAN));
      }
      writer.addDocument(doc);
    }
    writer.commit();
    reader = writer.getReader();
    searcher = newSearcher(reader);
    writer.close();
  }

  @Test
  public void testVectorSearchWithAlternatingDocuments() throws IOException {

    float[][] dataset = dataProvider.getDataset1();
    int topK = dataProvider.getTopK();
    float[] queryVector = dataProvider.getQueries(1)[0];

    Query query = new GPUKnnFloatVectorQuery(VECTOR_FIELD1, queryVector, topK, null, topK, 1);
    ScoreDoc[] hits = searcher.search(query, topK).scoreDocs;

    // Verify we get exactly topK results
    assertEquals("Should return exactly " + topK + " results", topK, hits.length);

    // Verify all returned documents have vectors (even-numbered IDs)
    for (ScoreDoc hit : hits) {
      Document doc = reader.storedFields().document(hit.doc);
      int id = Integer.parseInt(doc.get(ID_FIELD));
      assertEquals("All results should be even-numbered (have vectors)", 0, id % 2);
      log.log(Level.FINE, "Document ID: " + id + ", Score: " + hit.score);
    }

    // Verify the results match expected top-k
    List<Integer> expectedIds =
        generateExpectedTopK(topK, dataset, new float[][] {queryVector}).get(0);
    for (ScoreDoc hit : hits) {
      Document doc = reader.storedFields().document(hit.doc);
      int id = Integer.parseInt(doc.get(ID_FIELD));
      assertTrue("Result " + id + " should be in expected top-k results", expectedIds.contains(id));
    }

    log.log(Level.FINE, "Alternating document test passed with " + hits.length + " results");
  }

  @Test
  public void testVectorSearchWithFilterAndAlternatingDocuments() throws IOException {

    int datasetSize = dataProvider.getDatasetSize();
    int topK = dataProvider.getTopK();
    float[] queryVector = dataProvider.getQueries(1)[0];

    String randomEvenInRange = String.valueOf(random.nextInt(datasetSize / 2 + 1) * 2);
    log.log(Level.FINE, "Randomly chosen even value is: " + randomEvenInRange);
    Query filter = new TermQuery(new Term(ID_FIELD, randomEvenInRange));

    Query filteredQuery =
        new GPUKnnFloatVectorQuery(VECTOR_FIELD1, queryVector, topK, filter, topK, 1);
    ScoreDoc[] filteredHits = searcher.search(filteredQuery, topK).scoreDocs;

    // Should only get document (the only one that matches the filter and has a vector)
    assertEquals("Should return exactly 1 result", 1, filteredHits.length);
    Document doc = reader.storedFields().document(filteredHits[0].doc);
    assertEquals(
        "Should only return document " + randomEvenInRange, randomEvenInRange, doc.get(ID_FIELD));

    log.log(
        Level.FINE,
        "Filtered alternating document test passed with " + filteredHits.length + " results");
  }

  @AfterClass
  public static void afterClass() throws Exception {
    if (reader != null) reader.close();
    if (directory != null) directory.close();
    searcher = null;
    reader = null;
    directory = null;
  }
}
