/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.TestDataProvider.CATEGORY_FIELD;
import static com.nvidia.cuvs.lucene.TestDataProvider.ID_FIELD;
import static com.nvidia.cuvs.lucene.TestDataProvider.VECTOR_FIELD1;
import static com.nvidia.cuvs.lucene.TestUtils.createWriter;
import static com.nvidia.cuvs.lucene.TestUtils.createWriterConfig;
import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.isSupported;
import static org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.index.RandomIndexWriter;
import org.apache.lucene.tests.store.BaseDirectoryWrapper;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.LuceneTestCase.SuppressSysoutChecks;
import org.apache.lucene.tests.util.TestUtil;
import org.junit.BeforeClass;
import org.junit.Test;

@SuppressSysoutChecks(bugUrl = "")
public class TestCuVSDeletedDocuments extends LuceneTestCase {

  private static final Logger log = Logger.getLogger(TestCuVSDeletedDocuments.class.getName());
  private static Codec codec;
  private static Random random;
  private static TestDataProvider dataProvider;
  private static float deletionProbability;
  private static float vectorProbability;

  @BeforeClass
  public static void beforeClass() throws Exception {
    assumeTrue("cuVS not supported", isSupported());
    random = random();
    dataProvider = new TestDataProvider(random);
    codec = TestUtil.alwaysKnnVectorsFormat(new CuVS2510GPUVectorsFormat());
    deletionProbability = random.nextFloat() * 0.4f + 0.1f;
    vectorProbability = random.nextFloat() * 0.5f + 0.3f;

    log.log(
        Level.FINE,
        ", deletion probability: "
            + deletionProbability
            + ", vector probability: "
            + vectorProbability);
  }

  @Test
  public void testVectorSearchWithDeletedDocuments() throws IOException {

    int datasetSize = dataProvider.getDatasetSize();
    float[][] dataset = dataProvider.getDataset1();
    int topK = dataProvider.getTopK();
    float[] queryVector = dataProvider.getQueries(1)[0];

    try (BaseDirectoryWrapper directory = newDirectory()) {
      Set<Integer> deletedDocs = new HashSet<>();

      // Create index with all documents having vectors
      try (RandomIndexWriter writer = createWriter(random, directory, codec)) {
        // Add documents
        for (int i = 0; i < datasetSize; i++) {
          Document doc = new Document();
          doc.add(new StringField(ID_FIELD, String.valueOf(i), Field.Store.YES));
          doc.add(new KnnFloatVectorField(VECTOR_FIELD1, dataset[i], EUCLIDEAN));
          writer.addDocument(doc);
        }
        writer.commit();

        // Delete documents randomly based on probability
        for (int i = 0; i < datasetSize; i++) {
          if (random.nextFloat() < deletionProbability) {
            writer.deleteDocuments(new Term(ID_FIELD, String.valueOf(i)));
            deletedDocs.add(i);
          }
        }
        writer.commit();
      }

      log.log(Level.FINE, "Number of documents deleted: " + deletedDocs.size());

      // Search and verify deleted documents are not returned
      try (DirectoryReader reader = DirectoryReader.open(directory)) {
        IndexSearcher searcher = newSearcher(reader);

        Query query = new GPUKnnFloatVectorQuery(VECTOR_FIELD1, queryVector, topK, null, topK, 1);
        ScoreDoc[] hits = searcher.search(query, topK).scoreDocs;

        // Verify we got results
        assertTrue("Should have search results", hits.length > 0);

        // Verify no deleted documents in results
        for (ScoreDoc hit : hits) {
          Document doc = reader.storedFields().document(hit.doc);
          int id = Integer.parseInt(doc.get(ID_FIELD));
          assertFalse(
              "Deleted document " + id + " should not appear in results", deletedDocs.contains(id));
          log.log(Level.FINE, "Found non-deleted document: " + id + ", Score: " + hit.score);
        }

        // Verify deleted documents are truly deleted
        for (int deletedId : deletedDocs) {
          Query termQuery = new TermQuery(new Term(ID_FIELD, String.valueOf(deletedId)));
          TopDocs result = searcher.search(termQuery, 1);
          assertEquals(
              "Deleted document " + deletedId + " should not be found",
              0,
              result.totalHits.value());
        }
      }
    }
  }

  @Test
  public void testVectorSearchWithMixedDeletedAndMissingVectors() throws IOException {

    try (Directory directory = newDirectory()) {
      Set<Integer> docsWithoutVectors = new HashSet<>();
      Set<Integer> deletedDocs = new HashSet<>();
      int datasetSize = dataProvider.getDatasetSize();
      float[][] dataset = dataProvider.getDataset1();
      int topK = dataProvider.getTopK();
      float[] queryVector = dataProvider.getQueries(1)[0];

      // Create index with mixed documents
      try (RandomIndexWriter writer = createWriter(random, directory, codec)) {
        for (int i = 0; i < datasetSize; i++) {
          Document doc = new Document();
          doc.add(new StringField(ID_FIELD, String.valueOf(i), Field.Store.YES));
          // Randomly assign categories
          String category = random.nextBoolean() ? "A" : "B";
          doc.add(new StringField(CATEGORY_FIELD, category, Field.Store.YES));

          // Randomly decide whether to add vectors
          if (random.nextFloat() < vectorProbability) {
            doc.add(new KnnFloatVectorField(VECTOR_FIELD1, dataset[i], EUCLIDEAN));
          } else {
            docsWithoutVectors.add(i);
          }
          writer.addDocument(doc);
        }

        // Delete documents randomly
        for (int i = 0; i < datasetSize; i++) {
          if (random.nextFloat() < deletionProbability) {
            writer.deleteDocuments(new Term(ID_FIELD, String.valueOf(i)));
            deletedDocs.add(i);
          }
        }
        writer.commit();
      }

      log.log(Level.FINE, "Number of deleted documents: " + deletedDocs.size());

      // Test vector search behavior
      try (DirectoryReader reader = DirectoryReader.open(directory)) {
        IndexSearcher searcher = newSearcher(reader);

        Query query = new GPUKnnFloatVectorQuery(VECTOR_FIELD1, queryVector, topK, null, topK, 1);
        ScoreDoc[] hits = searcher.search(query, topK).scoreDocs;

        // Verify results
        for (ScoreDoc hit : hits) {
          Document doc = reader.storedFields().document(hit.doc);
          int id = Integer.parseInt(doc.get(ID_FIELD));
          assertFalse("Deleted document should not appear", deletedDocs.contains(id));
          assertFalse("Document without vector should not appear", docsWithoutVectors.contains(id));
          log.log(Level.FINE, "Found document with vector: " + id + ", Score: " + hit.score);
        }

        // Test filtered search with deletions
        Query filter = new TermQuery(new Term(CATEGORY_FIELD, "A"));
        Query filteredQuery =
            new GPUKnnFloatVectorQuery(VECTOR_FIELD1, queryVector, topK, filter, topK, 1);
        ScoreDoc[] filteredHits = searcher.search(filteredQuery, topK).scoreDocs;

        for (ScoreDoc hit : filteredHits) {
          Document doc = reader.storedFields().document(hit.doc);
          String category = doc.get(CATEGORY_FIELD);
          assertEquals("Should only match category A", "A", category);
          int id = Integer.parseInt(doc.get(ID_FIELD));
          assertFalse(
              "Deleted document should not appear in filtered results", deletedDocs.contains(id));
        }
      }
    }
  }

  @Test
  public void testVectorSearchAfterAllDocumentsDeleted() throws IOException {

    try (Directory directory = newDirectory()) {
      int datasetSize = dataProvider.getDatasetSize();
      float[][] dataset = dataProvider.getDataset1();
      int topK = dataProvider.getTopK();
      float[] queryVector = dataProvider.getQueries(1)[0];

      // Create and delete all documents
      try (IndexWriter writer = new IndexWriter(directory, createWriterConfig(random, codec))) {
        for (int i = 0; i < datasetSize; i++) {
          Document doc = new Document();
          doc.add(new StringField(ID_FIELD, String.valueOf(i), Field.Store.YES));
          doc.add(new KnnFloatVectorField(VECTOR_FIELD1, dataset[i], EUCLIDEAN));
          writer.addDocument(doc);
        }
        writer.commit();

        // Delete all documents
        for (int i = 0; i < datasetSize; i++) {
          writer.deleteDocuments(new Term(ID_FIELD, String.valueOf(i)));
        }
        writer.commit();
        writer.forceMerge(1); // Force merge to apply deletions
      }

      // Verify search returns no results
      try (DirectoryReader reader = DirectoryReader.open(directory)) {
        IndexSearcher searcher = newSearcher(reader);

        Query query = new GPUKnnFloatVectorQuery(VECTOR_FIELD1, queryVector, topK, null, topK, 1);
        TopDocs results = searcher.search(query, topK);

        assertEquals(
            "Should return no results when all documents are deleted",
            0,
            results.totalHits.value());
      }
    }
  }

  @Test
  public void testVectorSearchWithPartialDeletionAndReindexing() throws IOException {

    try (Directory directory = newDirectory()) {
      List<Integer> activeDocIds = new ArrayList<>();
      int datasetSize = dataProvider.getDatasetSize();
      float[][] dataset = dataProvider.getDataset1();
      int topK = dataProvider.getTopK();
      float[] queryVector = dataProvider.getQueries(1)[0];

      // Initial indexing
      try (IndexWriter writer = new IndexWriter(directory, createWriterConfig(random, codec))) {
        int initialDocs = datasetSize / 2 + random.nextInt(datasetSize / 4);
        for (int i = 0; i < initialDocs; i++) {
          Document doc = new Document();
          doc.add(new StringField(ID_FIELD, String.valueOf(i), Field.Store.YES));
          doc.add(new KnnFloatVectorField(VECTOR_FIELD1, dataset[i], EUCLIDEAN));
          writer.addDocument(doc);
          activeDocIds.add(i);
        }

        // Delete some documents randomly
        List<Integer> candidatesForDeletion = new ArrayList<>(activeDocIds);
        for (int docId : candidatesForDeletion) {
          if (random.nextFloat() < deletionProbability) {
            writer.deleteDocuments(new Term(ID_FIELD, String.valueOf(docId)));
            activeDocIds.remove(Integer.valueOf(docId));
          }
        }

        // Add new documents with higher IDs
        for (int i = initialDocs; i < datasetSize; i++) {
          Document doc = new Document();
          doc.add(new StringField(ID_FIELD, String.valueOf(i), Field.Store.YES));
          doc.add(new KnnFloatVectorField(VECTOR_FIELD1, dataset[i], EUCLIDEAN));
          writer.addDocument(doc);
          activeDocIds.add(i);
        }
        writer.commit();
      }

      // Verify search behavior after deletions and additions
      try (DirectoryReader reader = DirectoryReader.open(directory)) {
        IndexSearcher searcher = newSearcher(reader);

        Query query = new GPUKnnFloatVectorQuery(VECTOR_FIELD1, queryVector, topK, null, topK, 1);
        ScoreDoc[] hits = searcher.search(query, topK).scoreDocs;

        Set<Integer> resultIds = new HashSet<>();
        for (ScoreDoc hit : hits) {
          int id = Integer.parseInt(reader.storedFields().document(hit.doc).get(ID_FIELD));
          resultIds.add(id);
          assertTrue("Result should be from active documents", activeDocIds.contains(id));
        }

        log.log(
            Level.FINE,
            "Search returned "
                + hits.length
                + " results from "
                + activeDocIds.size()
                + " active documents");
      }
    }
  }
}
