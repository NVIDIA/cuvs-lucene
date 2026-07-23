/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.TestDataProvider.ID_FIELD;
import static com.nvidia.cuvs.lucene.TestDataProvider.VECTOR_FIELD1;
import static com.nvidia.cuvs.lucene.TestDataProvider.VECTOR_FIELD2;
import static com.nvidia.cuvs.lucene.TestUtils.createWriter;
import static com.nvidia.cuvs.lucene.TestUtils.generateExpectedTopK;
import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.isSupported;
import static org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN;

import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.UUID;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.apache.commons.io.FileUtils;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.FloatVectorValues;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.tests.index.RandomIndexWriter;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.LuceneTestCase.SuppressSysoutChecks;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

@SuppressSysoutChecks(bugUrl = "")
public class TestAcceleratedHNSW extends LuceneTestCase {

  private static final Logger log = Logger.getLogger(TestAcceleratedHNSW.class.getName());
  private static Random random;
  private static Path indexDirPath;
  private static String randomID;
  private static Codec codec;
  private static TestDataProvider dataProvider;

  @Before
  public void beforeTest() throws Exception {
    assumeTrue("cuVS not supported", isSupported());
    random = new Random();
    // Fixed seed so that we can validate against the same result.
    random = new Random(222);
    indexDirPath = Paths.get(UUID.randomUUID().toString());
    randomID = UUID.randomUUID().toString();
    dataProvider = new TestDataProvider(random);
    codec = new Lucene101AcceleratedHNSWCodec();
  }

  @Test
  public void testAcceleratedHNSW() throws Exception {
    // Indexing
    try (Directory indexDirectory = FSDirectory.open(indexDirPath);
        RandomIndexWriter indexWriter = createWriter(random, indexDirectory, codec)) {
      for (int i = 0; i < dataProvider.getDatasetSize(); i++) {
        Document document = new Document();
        document.add(new StringField(ID_FIELD, Integer.toString(i), Field.Store.YES));
        document.add(
            new KnnFloatVectorField(VECTOR_FIELD1, dataProvider.getDataset1()[i], EUCLIDEAN));
        document.add(
            new KnnFloatVectorField(VECTOR_FIELD2, dataProvider.getDataset2()[i], EUCLIDEAN));
        indexWriter.addDocument(document);
      }
      indexWriter.commit();
    }

    // Searching
    try (Directory indexDirectory = FSDirectory.open(indexDirPath);
        DirectoryReader reader = DirectoryReader.open(indexDirectory)) {

      int datasetSize = dataProvider.getDatasetSize();
      int dimensions = dataProvider.getDimensions();
      float[][] dataset = dataProvider.getDataset1();
      int topK = dataProvider.getTopK();
      float[] queryVector = dataProvider.getQueries(1)[0];

      int vectorCount = 0;
      for (LeafReaderContext leafReaderContext : reader.leaves()) {
        LeafReader leafReader = leafReaderContext.reader();
        FloatVectorValues knnValues = leafReader.getFloatVectorValues(VECTOR_FIELD1);
        assertNotNull(knnValues);
        log.log(
            Level.FINE,
            VECTOR_FIELD1
                + " field: "
                + knnValues.size()
                + " vectors, "
                + knnValues.dimension()
                + " dimensions");
        vectorCount += knnValues.size();
        assertTrue("Vector dimension mismatch", knnValues.dimension() == dimensions);
      }
      assertTrue("Dataset size mismatch", vectorCount == datasetSize);

      log.log(Level.FINE, "Testing vector search queries...");
      IndexSearcher searcher = new IndexSearcher(reader);

      log.log(Level.FINER, "Query vector: " + Arrays.toString(queryVector));

      KnnFloatVectorQuery query = new KnnFloatVectorQuery(VECTOR_FIELD1, queryVector, topK);
      TopDocs results = searcher.search(query, topK);

      log.log(Level.FINE, "Search results (" + results.totalHits + " total hits):");
      List<List<Integer>> expected =
          generateExpectedTopK(topK, dataset, new float[][] {queryVector});

      for (int i = 0; i < results.scoreDocs.length; i++) {
        ScoreDoc scoreDoc = results.scoreDocs[i];
        Document doc = searcher.storedFields().document(scoreDoc.doc);
        int id = Integer.valueOf(doc.get(ID_FIELD));
        log.log(
            Level.FINE,
            "  Rank "
                + (i + 1)
                + ": doc "
                + scoreDoc.doc
                + " (id="
                + id
                + "), score="
                + scoreDoc.score);
        assertTrue("Id: " + id + " expected but not found", expected.get(0).contains(id));
      }
      assertTrue("TopK results not returned", results.scoreDocs.length == topK);
    }
  }

  @Test
  public void testSingleVectorIndex() throws Exception {
    try (Directory indexDirectory = newDirectory()) {

      int dimensions = dataProvider.getDimensions();
      float[] queryVector = dataProvider.getQueries(1)[0];

      IndexWriterConfig config = new IndexWriterConfig().setCodec(codec).setUseCompoundFile(false);
      try (IndexWriter indexWriter = new IndexWriter(indexDirectory, config)) {
        Document document = new Document();
        document.add(new StringField(ID_FIELD, randomID, Field.Store.YES));
        document.add(new KnnFloatVectorField(VECTOR_FIELD1, queryVector, EUCLIDEAN));
        indexWriter.addDocument(document);
        indexWriter.commit();
      }

      // Verify the index can be opened and searched
      try (DirectoryReader reader = DirectoryReader.open(indexDirectory)) {
        assertEquals(1, reader.numDocs());
        LeafReader leafReader = getOnlyLeafReader(reader);
        FloatVectorValues knnValues = leafReader.getFloatVectorValues(VECTOR_FIELD1);
        assertNotNull(knnValues);
        assertEquals(1, knnValues.size());
        assertEquals(dimensions, knnValues.dimension());

        // Test search functionality
        IndexSearcher searcher = new IndexSearcher(reader);
        KnnFloatVectorQuery query = new KnnFloatVectorQuery(VECTOR_FIELD1, queryVector, 1);
        TopDocs results = searcher.search(query, 1);
        assertEquals(1, results.totalHits.value());
        assertEquals(1, results.scoreDocs.length);
        Document doc = reader.storedFields().document(results.scoreDocs[0].doc);
        assertEquals(randomID, doc.get(ID_FIELD));
      }
    }
  }

  @After
  public void afterTest() throws Exception {
    File indexDirPathFile = indexDirPath.toFile();
    if (indexDirPathFile.exists() && indexDirPathFile.isDirectory()) {
      FileUtils.deleteDirectory(indexDirPathFile);
    }
  }
}
