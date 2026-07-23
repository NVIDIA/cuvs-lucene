/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.TestDataProvider.ID_FIELD;
import static com.nvidia.cuvs.lucene.TestDataProvider.TEXT_FIELD;
import static com.nvidia.cuvs.lucene.TestDataProvider.VECTOR_FIELD1;
import static com.nvidia.cuvs.lucene.TestDataProvider.VECTOR_FIELD2;
import static com.nvidia.cuvs.lucene.TestUtils.createWriter;
import static com.nvidia.cuvs.lucene.TestUtils.generateExpectedTopK;
import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.isSupported;
import static org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN;

import java.io.IOException;
import java.util.Arrays;
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
import org.apache.lucene.search.KnnFloatVectorQuery;
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
public class TestAcceleratedHNSWRandomizedSearch extends LuceneTestCase {

  private static final Logger log =
      Logger.getLogger(TestAcceleratedHNSWRandomizedSearch.class.getName());
  private static Codec codec;
  private static IndexSearcher searcher;
  private static IndexReader reader;
  private static Directory directory;
  private static Random random;
  private static TestDataProvider dataProvider;

  @BeforeClass
  public static void beforeClass() throws Exception {
    assumeTrue("cuVS not supported so skipping these tests", isSupported());
    directory = newDirectory();
    random = random();
    dataProvider = new TestDataProvider(random);
    codec = TestUtil.alwaysKnnVectorsFormat(new Lucene99AcceleratedHNSWVectorsFormat());
    RandomIndexWriter writer = createWriter(random, directory, codec);

    int datasetSize = dataProvider.getDatasetSize();
    float[][] dataset = dataProvider.getDataset1();
    float[][] dataset2 = dataProvider.getDataset2();

    // Add documents
    for (int i = 0; i < datasetSize; i++) {
      Document doc = new Document();
      doc.add(new StringField(ID_FIELD, String.valueOf(i), Field.Store.YES));
      doc.add(newTextField(TEXT_FIELD, English.intToEnglish(i), Field.Store.YES));
      boolean skipVector = random.nextInt(10) < 4;
      if (!skipVector || datasetSize < 100) {
        doc.add(new KnnFloatVectorField(VECTOR_FIELD1, dataset[i], EUCLIDEAN));
        doc.add(new KnnFloatVectorField(VECTOR_FIELD2, dataset2[i], EUCLIDEAN));
      }
      writer.addDocument(doc);
    }
    writer.commit();
    reader = writer.getReader();
    searcher = newSearcher(reader);
    writer.close();
  }

  @Test
  public void testVectorSearch() throws IOException {

    float[][] dataset = dataProvider.getDataset1();
    int topK = dataProvider.getTopK();
    int numQueries = dataProvider.getNumQueries();
    float[][] queries = dataProvider.getQueries(numQueries);

    // Generate queries and expected results for each
    List<List<Integer>> expected = generateExpectedTopK(topK, dataset, queries);

    for (int i = 0; i < numQueries; i++) {
      log.log(Level.FINE, "Running query: " + (i + 1) + " of " + numQueries);
      Query query = new KnnFloatVectorQuery(VECTOR_FIELD1, queries[i], topK);

      // Perform search
      ScoreDoc[] hits = searcher.search(query, topK).scoreDocs;
      log.log(Level.FINE, "RESULTS: " + Arrays.toString(hits));
      log.log(Level.FINE, "EXPECTED: " + expected.get(i));

      // Iterate through the results and assert
      for (ScoreDoc hit : hits) {
        Document doc = reader.storedFields().document(hit.doc);
        int docId = Integer.parseInt(doc.get(ID_FIELD));
        log.log(Level.FINE, "\t" + doc.get(ID_FIELD) + ": " + hit.score);
        assertTrue("Result returned was not in topk*2: " + doc, expected.get(i).contains(docId));
      }
    }
  }

  @Test
  public void testVectorSearchWithFilter() throws IOException {
    // Find a document that has a vector by doing a search first

    int topK = dataProvider.getTopK();
    float[] queryVector = dataProvider.getQueries(1)[0];

    Query unfiltered = new KnnFloatVectorQuery(VECTOR_FIELD1, queryVector, 1);
    ScoreDoc[] unfilteredHits = searcher.search(unfiltered, 1).scoreDocs;

    assertTrue(
        "Need at least one document with vector for filtering test", unfilteredHits.length > 0);

    Document doc = reader.storedFields().document(unfilteredHits[0].doc);
    String targetDocId = doc.get(ID_FIELD);

    // Create a filter that matches only the document we know has a vector
    Query filter = new TermQuery(new Term(ID_FIELD, targetDocId));

    // Test the new constructor with filter
    Query filteredQuery = new KnnFloatVectorQuery(VECTOR_FIELD1, queryVector, topK, filter);

    ScoreDoc[] filteredHits = searcher.search(filteredQuery, topK).scoreDocs;

    // Ensure we got some results
    assertTrue("Should have at least one result", filteredHits.length > 0);

    // Verify that all results match the filter
    for (ScoreDoc hit : filteredHits) {
      String docId = reader.storedFields().document(hit.doc).get(ID_FIELD);
      assertEquals("All results should match the filter", targetDocId, docId);
    }

    log.log(Level.FINE, "Prefiltering test passed with " + filteredHits.length + " results");
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
