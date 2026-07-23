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
import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.setCuVSResourcesInstance;
import static com.nvidia.cuvs.lucene.Utils.cuVSResourcesOrNull;
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
public class TestAcceleratedHNSWFallback extends LuceneTestCase {

  private static final Logger log = Logger.getLogger(TestAcceleratedHNSWFallback.class.getName());
  private static Random random;
  private static Path indexDirPath;
  private static Codec codec;
  private static TestDataProvider dataProvider;

  @Before
  public void beforeTest() throws Exception {
    assumeTrue("cuVS not supported so skipping these tests", isSupported());
    // Set resources to null to simulate that cuVS is not supported.
    setCuVSResourcesInstance(null);
    random = new Random();
    dataProvider = new TestDataProvider(random);
    indexDirPath = Paths.get(UUID.randomUUID().toString());
    codec = new Lucene101AcceleratedHNSWCodec();
  }

  @Test
  public void testAcceleratedHNSWFallback() throws Exception {

    int datasetSize = dataProvider.getDatasetSize();
    int dimensions = dataProvider.getDimensions();
    float[][] dataset = dataProvider.getDataset1();
    float[][] dataset2 = dataProvider.getDataset2();
    int topK = dataProvider.getTopK();
    float[] queryVector = dataProvider.getQueries(1)[0];

    // Indexing
    try (Directory indexDirectory = FSDirectory.open(indexDirPath);
        RandomIndexWriter indexWriter = createWriter(random, indexDirectory, codec)) {
      for (int i = 0; i < datasetSize; i++) {
        Document document = new Document();
        document.add(new StringField(ID_FIELD, Integer.toString(i), Field.Store.YES));
        document.add(new KnnFloatVectorField(VECTOR_FIELD1, dataset[i], EUCLIDEAN));
        document.add(new KnnFloatVectorField(VECTOR_FIELD2, dataset2[i], EUCLIDEAN));
        indexWriter.addDocument(document);
      }
      indexWriter.commit();
    }

    // Searching
    try (Directory indexDirectory = FSDirectory.open(indexDirPath);
        DirectoryReader reader = DirectoryReader.open(indexDirectory)) {
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

  @After
  public void afterTest() throws Exception {
    // Reset resources for other tests to work
    setCuVSResourcesInstance(cuVSResourcesOrNull());

    File indexDirPathFile = indexDirPath.toFile();
    if (indexDirPathFile.exists() && indexDirPathFile.isDirectory()) {
      FileUtils.deleteDirectory(indexDirPathFile);
    }
  }
}
