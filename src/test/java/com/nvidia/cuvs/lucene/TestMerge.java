/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.TestDataProvider.ID_FIELD;
import static com.nvidia.cuvs.lucene.TestDataProvider.VECTOR_FIELD1;
import static com.nvidia.cuvs.lucene.TestUtils.generateRandomText;
import static com.nvidia.cuvs.lucene.ThreadLocalCuVSResourcesProvider.isSupported;
import static org.apache.lucene.index.VectorSimilarityFunction.EUCLIDEAN;
import static org.apache.lucene.tests.util.TestUtil.alwaysKnnVectorsFormat;

import com.carrotsearch.randomizedtesting.annotations.Name;
import com.carrotsearch.randomizedtesting.annotations.ParametersFactory;
import com.nvidia.cuvs.CagraIndexParams.CagraGraphBuildAlgo;
import com.nvidia.cuvs.lucene.CuVS2510GPUVectorsWriter.IndexType;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
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
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.SortedDocValuesField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.SortedDocValues;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.TieredMergePolicy;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.Sort;
import org.apache.lucene.search.SortField;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.util.LuceneTestCase;
import org.apache.lucene.tests.util.LuceneTestCase.SuppressSysoutChecks;
import org.apache.lucene.util.BytesRef;
import org.junit.After;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

/**
 * Comprehensive tests for merge functionality with CuVS indexes.
 * Tests merge operations across different index types including brute force,
 * CAGRA, and combined index configurations to ensure proper vector handling
 * and search functionality after segment merging.
 */
@SuppressSysoutChecks(bugUrl = "")
public class TestMerge extends LuceneTestCase {

  private static final Logger log = Logger.getLogger(TestMerge.class.getName());
  private static Random random;
  private static TestDataProvider dataProvider;
  private static Directory directory;
  private static Codec codec;

  private static CagraGraphBuildAlgo cagraGraphBuildAlgo;

  public TestMerge(@Name("cagraGraphBuildAlgo") CagraGraphBuildAlgo cagraGraphBuildAlgo) {
    TestMerge.cagraGraphBuildAlgo = cagraGraphBuildAlgo;
  }

  @ParametersFactory
  public static List<Object[]> parameters() {
    return Arrays.asList(
        new Object[][] {{CagraGraphBuildAlgo.NN_DESCENT}, {CagraGraphBuildAlgo.IVF_PQ}});
  }

  @BeforeClass
  public static void beforeClass() {
    assumeTrue("cuVS is not supported", isSupported());
    random = random();
    codec = alwaysKnnVectorsFormat(new CuVS2510GPUVectorsFormat());
  }

  @Before
  public void setUp() throws Exception {
    super.setUp();
    directory = newDirectory();
    dataProvider = new TestDataProvider(random);
  }

  @After
  public void tearDown() throws Exception {
    if (directory != null) {
      directory.close();
    }
    super.tearDown();
  }

  /**
   *  Test merging many documents across multiple segments
   **/
  @Test
  public void testMergeManyDocumentsMultipleSegments() throws IOException {
    log.log(
        Level.FINE,
        "Starting testMergeManyDocumentsMultipleSegments with CagraGraphBuildAlgo: "
            + cagraGraphBuildAlgo);

    // Randomize configuration parameters
    int maxBufferedDocs = dataProvider.getRandom(5, 16);
    int totalBatches = dataProvider.getRandom(8, 16);
    int docsPerBatch = dataProvider.getRandom(15, 25);
    int totalDocuments = totalBatches * docsPerBatch;
    double vectorProbability = dataProvider.getRandom(0.6, 0.8);

    log.log(
        Level.FINE,
        "Randomized parameters: maxBufferedDocs="
            + maxBufferedDocs
            + ", totalBatches="
            + totalBatches
            + ", docsPerBatch="
            + docsPerBatch
            + ", totalDocuments="
            + totalDocuments
            + ", vectorProbability="
            + vectorProbability);

    GPUSearchParams params =
        new GPUSearchParams.Builder().withCagraGraphBuildAlgo(cagraGraphBuildAlgo).build();

    CuVS2510GPUVectorsFormat format = new CuVS2510GPUVectorsFormat(params);

    IndexWriterConfig config =
        new IndexWriterConfig()
            .setCodec(alwaysKnnVectorsFormat(format))
            .setMaxBufferedDocs(maxBufferedDocs)
            .setRAMBufferSizeMB(IndexWriterConfig.DISABLE_AUTO_FLUSH);

    List<Integer> expectedDocIds = new ArrayList<>();
    int documentsWithVectors = 0;

    try (IndexWriter writer = new IndexWriter(directory, config)) {
      // Add documents in multiple batches to create many segments
      for (int batch = 0; batch < totalBatches; batch++) {
        for (int i = 0; i < docsPerBatch; i++) {
          int docId = batch * docsPerBatch + i;
          Document doc = new Document();
          doc.add(new StringField(ID_FIELD, String.valueOf(docId), Field.Store.YES));

          // Randomly decide if document has vector
          if (random().nextDouble() < vectorProbability) {
            float[] vector = dataProvider.getQueries(1)[0];
            doc.add(new KnnFloatVectorField(VECTOR_FIELD1, vector, EUCLIDEAN));
            expectedDocIds.add(docId);
            documentsWithVectors++;
          }
          writer.addDocument(doc);
        }
        writer.commit();
      }

      int documentsWithoutVectors = totalDocuments - documentsWithVectors;
      log.log(
          Level.FINE, "Created " + totalDocuments + " documents in " + totalBatches + " segments");
      log.log(Level.FINE, "Documents with vectors: " + documentsWithVectors);
      log.log(Level.FINE, "Documents without vectors: " + documentsWithoutVectors);

      writer.forceMerge(1);
      log.log(Level.FINE, "Forced merge to single segment completed");
    }

    // Verify the merged index
    try (DirectoryReader reader = DirectoryReader.open(directory)) {
      List<LeafReaderContext> leaves = reader.leaves();
      assertEquals("Should have exactly one segment after merge", 1, leaves.size());
      LeafReader leafReader = leaves.get(0).reader();
      assertEquals("Total documents should match", totalDocuments, leafReader.maxDoc());

      IndexSearcher searcher = new IndexSearcher(reader);
      float[] queryVector = dataProvider.getQueries(1)[0];
      int topK = dataProvider.getTopK();

      KnnFloatVectorQuery query = new KnnFloatVectorQuery(VECTOR_FIELD1, queryVector, topK);
      TopDocs results = searcher.search(query, topK);

      assertTrue("Should find some results after merge", results.scoreDocs.length > 0);
      assertTrue("Should find some results", results.scoreDocs.length <= documentsWithVectors);

      log.log(
          Level.FINE,
          "Vector search returned "
              + results.scoreDocs.length
              + " results out of "
              + documentsWithVectors
              + " documents with vectors");

      // Verify all returned documents have valid IDs
      for (ScoreDoc scoreDoc : results.scoreDocs) {
        Document doc = searcher.storedFields().document(scoreDoc.doc);
        int docId = Integer.parseInt(doc.get(ID_FIELD));
        assertTrue("Document ID should be valid", expectedDocIds.contains(docId));
      }
    }
  }

  /**
   * Test merging with index sorting enabled using SortingMergePolicy
   **/
  @Test
  public void testMergeWithIndexSortingStringField() throws IOException {
    log.log(
        Level.FINE,
        "Starting testMergeWithIndexSorting with text-based sorting with CagraGraphBuildAlgo: "
            + cagraGraphBuildAlgo);

    // Randomize sort field type
    final String SORT_FIELD_NAME = "text_sort_key";
    final String ORIGINAL_ORDER = "original_order";
    // Configure index sorting by a randomized field
    Sort indexSort = new Sort(new SortField(SORT_FIELD_NAME, SortField.Type.STRING));

    // Randomize merge policy parameters
    TieredMergePolicy mergePolicy = new TieredMergePolicy();
    mergePolicy.setMaxMergedSegmentMB(dataProvider.getRandom(128, 385));
    mergePolicy.setSegmentsPerTier(dataProvider.getRandom(3, 7));

    // Randomize writer configuration parameters
    int maxBufferedDocs = dataProvider.getRandom(10, 26);
    int totalDocuments = dataProvider.getRandom(80, 161);
    int segmentSize = dataProvider.getRandom(15, 26);
    double vectorProbability = dataProvider.getRandom(0.65, 0.91);

    log.log(
        Level.FINE,
        "Randomized config: maxBufferedDocs="
            + maxBufferedDocs
            + ", totalDocuments="
            + totalDocuments
            + ", segmentSize="
            + segmentSize
            + ", vectorProbability="
            + vectorProbability);

    GPUSearchParams params =
        new GPUSearchParams.Builder().withCagraGraphBuildAlgo(cagraGraphBuildAlgo).build();

    CuVS2510GPUVectorsFormat format = new CuVS2510GPUVectorsFormat(params);

    IndexWriterConfig config =
        new IndexWriterConfig()
            .setCodec(alwaysKnnVectorsFormat(format))
            .setIndexSort(indexSort) // This automatically enables sorting during merges
            .setMergePolicy(mergePolicy)
            .setMaxBufferedDocs(maxBufferedDocs)
            .setRAMBufferSizeMB(IndexWriterConfig.DISABLE_AUTO_FLUSH);

    try (IndexWriter writer = new IndexWriter(directory, config)) {
      // Create documents with randomized sort keys
      int numDocsWithVectors = 0;
      for (int i = 0; i < totalDocuments; i++) {

        Document doc = new Document();
        doc.add(new StringField(ID_FIELD, String.valueOf(i), Field.Store.YES));
        doc.add(new StringField(ORIGINAL_ORDER, String.valueOf(i), Field.Store.YES));

        String textSortKey = generateRandomText(random, dataProvider.getRandom(4, 21));
        doc.add(new SortedDocValuesField(SORT_FIELD_NAME, new BytesRef(textSortKey)));
        doc.add(new StringField(SORT_FIELD_NAME + "_stored", textSortKey, Field.Store.YES));

        if (random.nextDouble() < vectorProbability) {
          float[] vector = dataProvider.getQueries(1)[0];
          doc.add(new KnnFloatVectorField(VECTOR_FIELD1, vector, EUCLIDEAN));
          numDocsWithVectors++;
        }

        writer.addDocument(doc);

        // Commit based on randomized segment size
        if ((i + 1) % segmentSize == 0) {
          writer.commit();
          log.log(
              Level.FINE,
              "Committed segment "
                  + ((i + 1) / segmentSize)
                  + " with "
                  + (i + 1)
                  + " total documents");
        }
      }

      log.log(
          Level.FINE,
          "Number of documents with vectors is: "
              + numDocsWithVectors
              + " out of a total of "
              + totalDocuments
              + " documents");

      // Force merge with sorting - this will use the sorting merge policy
      writer.forceMerge(1);
      log.log(Level.FINE, "Forced merge with text-based sorting completed");
    }

    // Verify the merged and sorted index
    try (DirectoryReader reader = DirectoryReader.open(directory)) {
      List<LeafReaderContext> leaves = reader.leaves();
      assertEquals("Should have exactly one segment after merge", 1, leaves.size());
      LeafReader leafReader = leaves.get(0).reader();
      assertEquals("Total documents should match", totalDocuments, leafReader.maxDoc());

      String previousSortKey = "";
      SortedDocValues sortedValues = leafReader.getSortedDocValues(SORT_FIELD_NAME);

      for (int docId = 0; docId < leafReader.maxDoc(); docId++) {
        String currentSortKey = "";
        if (sortedValues != null && sortedValues.advanceExact(docId)) {
          currentSortKey = sortedValues.lookupOrd(sortedValues.ordValue()).utf8ToString();
        }

        assertTrue(
            "Documents should be sorted by "
                + SORT_FIELD_NAME
                + ": '"
                + previousSortKey
                + "' should be <= '"
                + currentSortKey
                + "'",
            previousSortKey.compareTo(currentSortKey) <= 0);
        previousSortKey = currentSortKey;
      }

      // Count total vectors by checking if vector field exists and has values
      var vectorValues = leafReader.getFloatVectorValues(VECTOR_FIELD1);
      int documentsWithVectors = vectorValues != null ? vectorValues.size() : 0;

      log.log(
          Level.FINE,
          "Found " + documentsWithVectors + " documents with vectors after sorted merge");

      // Test vector search on sorted index
      if (documentsWithVectors > 0) {
        IndexSearcher searcher = new IndexSearcher(reader);
        float[] queryVector = dataProvider.getQueries(1)[0];

        KnnFloatVectorQuery query =
            new KnnFloatVectorQuery(VECTOR_FIELD1, queryVector, Math.min(10, documentsWithVectors));
        TopDocs results = searcher.search(query, 10);

        assertTrue("Should find results in sorted index", results.scoreDocs.length > 0);
        log.log(
            Level.FINE,
            "Vector search on sorted index returned " + results.scoreDocs.length + " results");

        // Verify that returned documents maintain sort order if we check their sort keys
        log.log(Level.FINE, "Verifying vector search results maintain sorting consistency...");
        for (int i = 0; i < Math.min(5, results.scoreDocs.length); i++) {
          ScoreDoc scoreDoc = results.scoreDocs[i];
          Document doc = searcher.storedFields().document(scoreDoc.doc);
          String originalOrder = doc.get(ORIGINAL_ORDER);
          String sortKey = doc.get(SORT_FIELD_NAME + "_stored");
          log.log(
              Level.FINE,
              "Result "
                  + i
                  + ": DocId="
                  + scoreDoc.doc
                  + ", OriginalOrder="
                  + originalOrder
                  + ", SortKey='"
                  + sortKey
                  + "', Score="
                  + scoreDoc.score);
        }
      }
    }
  }

  /**
   * Test merging with index sorting enabled using SortingMergePolicy
   **/
  @Test
  public void testMergeWithIndexSortingLongField() throws IOException {
    final String SORT_FIELD_NAME = "numeric_sort_key";
    final String ORIGINAL_ORDER = "original_order";
    Sort indexSort = new Sort(new SortField(SORT_FIELD_NAME, SortField.Type.LONG));

    // Randomize merge policy parameters
    TieredMergePolicy mergePolicy = new TieredMergePolicy();
    mergePolicy.setMaxMergedSegmentMB(dataProvider.getRandom(128, 385));
    mergePolicy.setSegmentsPerTier(dataProvider.getRandom(3, 7));

    // Randomize writer configuration parameters
    int maxBufferedDocs = dataProvider.getRandom(10, 26);
    int totalDocuments = dataProvider.getRandom(80, 161);
    int segmentSize = dataProvider.getRandom(15, 26);
    double vectorProbability = dataProvider.getRandom(0.65, 0.91);

    log.log(
        Level.FINE,
        "Randomized config: maxBufferedDocs="
            + maxBufferedDocs
            + ", totalDocuments="
            + totalDocuments
            + ", segmentSize="
            + segmentSize
            + ", vectorProbability="
            + vectorProbability);

    IndexWriterConfig config =
        new IndexWriterConfig()
            .setCodec(codec)
            .setIndexSort(indexSort)
            .setMergePolicy(mergePolicy)
            .setMaxBufferedDocs(maxBufferedDocs)
            .setRAMBufferSizeMB(IndexWriterConfig.DISABLE_AUTO_FLUSH);

    try (IndexWriter writer = new IndexWriter(directory, config)) {
      int numDocsWithVectors = 0;
      for (int i = 0; i < totalDocuments; i++) {

        Document doc = new Document();
        doc.add(new StringField(ID_FIELD, String.valueOf(i), Field.Store.YES));
        doc.add(new StringField(ORIGINAL_ORDER, String.valueOf(i), Field.Store.YES));

        long numericSortKey = random.nextLong() % 100000;
        doc.add(new NumericDocValuesField(SORT_FIELD_NAME, numericSortKey));
        doc.add(
            new StringField(
                SORT_FIELD_NAME + "_stored", String.valueOf(numericSortKey), Field.Store.YES));

        if (random.nextDouble() < vectorProbability) {
          float[] vector = dataProvider.getQueries(1)[0];
          doc.add(new KnnFloatVectorField(VECTOR_FIELD1, vector, EUCLIDEAN));
          numDocsWithVectors++;
        }

        writer.addDocument(doc);

        // Commit based on randomized segment size
        if ((i + 1) % segmentSize == 0) {
          writer.commit();
          log.log(
              Level.FINE,
              "Committed segment "
                  + ((i + 1) / segmentSize)
                  + " with "
                  + (i + 1)
                  + " total documents");
        }
      }

      log.log(
          Level.FINE,
          "Number of documents with vectors is: "
              + numDocsWithVectors
              + " out of a total of "
              + totalDocuments
              + " documents");

      // Force merge with sorting - this will use the sorting merge policy
      writer.forceMerge(1);
      log.log(Level.FINE, "Forced merge with text-based sorting completed");
    }

    // Verify the merged and sorted index
    try (DirectoryReader reader = DirectoryReader.open(directory)) {
      List<LeafReaderContext> leaves = reader.leaves();
      assertEquals("Should have exactly one segment after merge", 1, leaves.size());
      LeafReader leafReader = leaves.get(0).reader();
      assertEquals("Total documents should match", totalDocuments, leafReader.maxDoc());

      // Verify numeric-based sorting
      long previousSortKey = Long.MIN_VALUE;
      var numericValues = leafReader.getNumericDocValues(SORT_FIELD_NAME);

      for (int docId = 0; docId < leafReader.maxDoc(); docId++) {
        long currentSortKey = Long.MIN_VALUE;
        if (numericValues != null && numericValues.advanceExact(docId)) {
          currentSortKey = numericValues.longValue();
        }

        assertTrue(
            "Documents should be sorted by "
                + SORT_FIELD_NAME
                + ": "
                + previousSortKey
                + " should be <= "
                + currentSortKey,
            previousSortKey <= currentSortKey);
        previousSortKey = currentSortKey;

        // Log first 10 documents to verify sorting
        if (docId < 10) {
          IndexSearcher searcher = new IndexSearcher(reader);
          String originalOrder = searcher.storedFields().document(docId).get(ORIGINAL_ORDER);
          log.log(
              Level.FINE,
              "DocId: "
                  + docId
                  + ", OriginalOrder: "
                  + originalOrder
                  + ", SortKey: "
                  + currentSortKey);
        }
      }

      // Count total vectors by checking if vector field exists and has values
      var vectorValues = leafReader.getFloatVectorValues(VECTOR_FIELD1);
      int documentsWithVectors = vectorValues != null ? vectorValues.size() : 0;

      log.log(
          Level.FINE,
          "Found " + documentsWithVectors + " documents with vectors after sorted merge");

      // Test vector search on sorted index
      if (documentsWithVectors > 0) {
        IndexSearcher searcher = new IndexSearcher(reader);
        float[] queryVector = dataProvider.getQueries(1)[0];

        KnnFloatVectorQuery query =
            new KnnFloatVectorQuery(VECTOR_FIELD1, queryVector, Math.min(10, documentsWithVectors));
        TopDocs results = searcher.search(query, 10);

        assertTrue("Should find results in sorted index", results.scoreDocs.length > 0);
        log.log(
            Level.FINE,
            "Vector search on sorted index returned " + results.scoreDocs.length + " results");

        // Verify that returned documents maintain sort order if we check their sort keys
        log.log(Level.FINE, "Verifying vector search results maintain sorting consistency...");
        for (int i = 0; i < Math.min(5, results.scoreDocs.length); i++) {
          ScoreDoc scoreDoc = results.scoreDocs[i];
          Document doc = searcher.storedFields().document(scoreDoc.doc);
          String originalOrder = doc.get(ORIGINAL_ORDER);
          String sortKey = doc.get(SORT_FIELD_NAME + "_stored");
          log.log(
              Level.FINE,
              "Result "
                  + i
                  + ": DocId="
                  + scoreDoc.doc
                  + ", OriginalOrder="
                  + originalOrder
                  + ", SortKey='"
                  + sortKey
                  + "', Score="
                  + scoreDoc.score);
        }
      }
    }
  }

  /**
   * Test merging segments with various patterns of missing vectors
   **/
  @Test
  public void testMergeWithMissingVectors() throws IOException {
    log.log(
        Level.FINE,
        "Starting testMergeWithMissingVectors with CagraGraphBuildAlgo: " + cagraGraphBuildAlgo);

    // Randomize configuration
    int maxBufferedDocs = 10 + random().nextInt(11); // 10-20 docs per buffer
    int numSegments = dataProvider.getRandom(3, 13);
    log.log(Level.FINE, "Randomized parameters: numSegments=" + numSegments);

    GPUSearchParams params =
        new GPUSearchParams.Builder().withCagraGraphBuildAlgo(cagraGraphBuildAlgo).build();

    CuVS2510GPUVectorsFormat format = new CuVS2510GPUVectorsFormat(params);

    IndexWriterConfig config =
        new IndexWriterConfig()
            .setCodec(alwaysKnnVectorsFormat(format))
            .setMaxBufferedDocs(maxBufferedDocs)
            .setRAMBufferSizeMB(IndexWriterConfig.DISABLE_AUTO_FLUSH);

    log.log(
        Level.FINE,
        "Randomized parameters: maxBufferedDocs="
            + maxBufferedDocs
            + ", numSegments="
            + numSegments);

    int totalExpectedVectors = 0;
    int totalDocuments = 0;

    try (IndexWriter writer = new IndexWriter(directory, config)) {
      for (int seg = 0; seg < numSegments; seg++) {
        int docsInSegment = dataProvider.getRandom(15, 100);
        double vectorProbability = dataProvider.getRandom(0.1, 0.6);
        int segmentVectorCount = 0;

        for (int i = 0; i < docsInSegment; i++) {
          Document doc = new Document();
          doc.add(new StringField(ID_FIELD, String.valueOf(i), Field.Store.YES));

          // Randomly add vector based on segment's probability
          if (random.nextDouble() < vectorProbability) {
            float[] vector = dataProvider.getQueries(1)[0];
            doc.add(new KnnFloatVectorField(VECTOR_FIELD1, vector, EUCLIDEAN));
            segmentVectorCount++;
          }
          writer.addDocument(doc);
        }

        writer.commit();
        totalDocuments += docsInSegment;
        totalExpectedVectors += segmentVectorCount;

        log.log(
            Level.FINE,
            "Created segment "
                + seg
                + ": "
                + docsInSegment
                + " documents, "
                + segmentVectorCount
                + " with vectors (probability: "
                + vectorProbability
                + ")");
      }

      // Force merge all segments
      writer.forceMerge(1);
      log.log(Level.FINE, "Forced merge of " + numSegments + " segments completed");
    }

    // Verify the merged index handles missing vectors correctly
    try (DirectoryReader reader = DirectoryReader.open(directory)) {
      List<LeafReaderContext> leaves = reader.leaves();
      assertEquals("Should have exactly one segment after merge", 1, leaves.size());
      LeafReader leafReader = leaves.get(0).reader();
      assertEquals("Total documents should match", totalDocuments, leafReader.maxDoc());

      // Count actual vectors in merged index
      var vectorValues = leafReader.getFloatVectorValues(VECTOR_FIELD1);
      int actualVectorCount = vectorValues != null ? vectorValues.size() : 0;

      log.log(
          Level.FINE,
          "Total documents: "
              + totalDocuments
              + ", Expected vectors: "
              + totalExpectedVectors
              + ", Actual vectors: "
              + actualVectorCount);

      assertEquals("Vector count should match expected", totalExpectedVectors, actualVectorCount);

      // Test vector search if we have vectors
      if (actualVectorCount > 0) {
        IndexSearcher searcher = new IndexSearcher(reader);
        float[] queryVector = dataProvider.getQueries(1)[0];
        int topK = Math.min(dataProvider.getTopK(), actualVectorCount);
        KnnFloatVectorQuery vectorQuery = new KnnFloatVectorQuery(VECTOR_FIELD1, queryVector, topK);
        TopDocs vectorResults = searcher.search(vectorQuery, topK);

        int numResults = vectorResults.scoreDocs.length;
        assertTrue("Should find some vector results", numResults > 0);
        assertTrue("Should not find more vectors than exist", numResults <= actualVectorCount);

        log.log(
            Level.FINE,
            "Found " + numResults + " vector results out of " + actualVectorCount + " available");
        assertEquals("Search should return exactly topK results", topK, numResults);
      } else {
        log.log(Level.FINE, "No vectors in merged index - skipping vector search");
      }
    }
  }

  /**
   *  Test merge behavior with document deletions
   **/
  @Test
  public void testMergeWithDeletions() throws IOException {
    int numSegments = dataProvider.getRandom(3, 7);
    int docsPerSegment = dataProvider.getRandom(20, 41);
    int maxBufferedDocs = dataProvider.getRandom(8, 17);
    double vectorProbability = dataProvider.getRandom(0.7, 0.95);
    double deletionProbability = dataProvider.getRandom(0.2, 0.5);

    log.log(
        Level.FINE,
        "Randomized parameters: numSegments="
            + numSegments
            + ", docsPerSegment="
            + docsPerSegment
            + ", vectorProbability="
            + vectorProbability
            + ", deletionProbability="
            + deletionProbability);

    GPUSearchParams params =
        new GPUSearchParams.Builder().withCagraGraphBuildAlgo(cagraGraphBuildAlgo).build();

    CuVS2510GPUVectorsFormat format = new CuVS2510GPUVectorsFormat(params);

    IndexWriterConfig config =
        new IndexWriterConfig()
            .setCodec(alwaysKnnVectorsFormat(format))
            .setMaxBufferedDocs(maxBufferedDocs)
            .setRAMBufferSizeMB(IndexWriterConfig.DISABLE_AUTO_FLUSH);

    List<Integer> expectedRemainingDocs = new ArrayList<>();
    List<Integer> deletedDocs = new ArrayList<>();
    int totalDocuments = numSegments * docsPerSegment;
    int numDocsWithVectors = 0;

    try (IndexWriter writer = new IndexWriter(directory, config)) {
      // Create multiple segments with documents
      for (int seg = 0; seg < numSegments; seg++) {
        for (int i = 0; i < docsPerSegment; i++) {
          int docId = seg * docsPerSegment + i;
          Document doc = new Document();
          doc.add(new StringField(ID_FIELD, String.valueOf(docId), Field.Store.YES));

          // Randomly add vectors
          if (random.nextDouble() < vectorProbability) {
            float[] vector = dataProvider.getQueries(1)[0];
            doc.add(new KnnFloatVectorField(VECTOR_FIELD1, vector, EUCLIDEAN));
            numDocsWithVectors++;
          }
          writer.addDocument(doc);
        }
        writer.commit();
      }

      log.log(
          Level.FINE,
          "Created "
              + numSegments
              + " segments with "
              + docsPerSegment
              + " documents each ("
              + totalDocuments
              + " total)");
      log.log(Level.FINE, "Number of docs with vectors: " + numDocsWithVectors);

      // Delete documents randomly and track which ones are deleted
      int deletedCount = 0;
      for (int docId = 0; docId < totalDocuments; docId++) {
        if (random.nextDouble() < deletionProbability) {
          writer.deleteDocuments(new Term(ID_FIELD, String.valueOf(docId)));
          deletedDocs.add(docId);
          deletedCount++;
        } else {
          expectedRemainingDocs.add(docId);
        }
      }

      log.log(
          Level.FINE,
          "Deleted "
              + deletedCount
              + " documents ("
              + (100.0 * deletedCount / totalDocuments)
              + "%), remaining: "
              + expectedRemainingDocs.size());

      writer.commit();

      // Force merge to apply deletions
      writer.forceMerge(1);
      log.log(Level.FINE, "Forced merge with deletions completed");
    }

    // Verify the merged index correctly handles deletions
    try (DirectoryReader reader = DirectoryReader.open(directory)) {
      List<LeafReaderContext> leaves = reader.leaves();
      assertEquals("Should have exactly one segment after merge", 1, leaves.size());
      LeafReader leafReader = leaves.get(0).reader();
      int expectedRemaining = expectedRemainingDocs.size();
      assertEquals(
          "Should have correct number of documents after deletions",
          expectedRemaining,
          leafReader.maxDoc());

      // Verify that deleted documents are not present
      IndexSearcher searcher = new IndexSearcher(reader);

      // Test that we can find expected remaining documents
      for (int i = 0; i < Math.min(10, expectedRemainingDocs.size()); i++) {
        int docId = expectedRemainingDocs.get(i);
        TopDocs result =
            searcher.search(new TermQuery(new Term(ID_FIELD, String.valueOf(docId))), 1);
        assertEquals("Should find remaining document " + docId, 1, (int) result.totalHits.value());
      }

      // Test that actually deleted documents are not found
      int deletedDocsToCheck = Math.min(10, deletedDocs.size());
      for (int i = 0; i < deletedDocsToCheck; i++) {
        int docId = deletedDocs.get(i);
        TopDocs result =
            searcher.search(new TermQuery(new Term(ID_FIELD, String.valueOf(docId))), 1);
        assertEquals(
            "Should not find deleted document " + docId, 0, (int) result.totalHits.value());
      }

      // Test vector search works after deletions
      float[] queryVector = dataProvider.getQueries(1)[0];
      int topK = Math.min(1, numDocsWithVectors);
      KnnFloatVectorQuery vectorQuery = new KnnFloatVectorQuery(VECTOR_FIELD1, queryVector, topK);
      TopDocs vectorResults = searcher.search(vectorQuery, topK);
      int numResults = vectorResults.scoreDocs.length;
      assertTrue("Should find some vector results after deletions", numResults > 0);
      assertEquals("Search should return exactly topK documents", numResults, topK);
      log.log(Level.FINE, "Found " + numResults + " vector results after deletions");
    }
  }

  /**
   * Test merging segments for {@link IndexType#BRUTE_FORCE}
   * */
  @Test
  public void testMergeBruteForceIndex() throws IOException {
    log.log(
        Level.FINE,
        "Starting testMergeBruteForceIndex with CagraGraphBuildAlgo: " + cagraGraphBuildAlgo);

    // Randomize configuration parameters
    int numSegments = dataProvider.getRandom(3, 10);
    int docsPerSegment = dataProvider.getRandom(20, 100);
    double vectorProbability = dataProvider.getRandom(0.2, 0.7);
    int maxBufferedDocs = dataProvider.getRandom(8, 17);

    log.log(
        Level.FINE,
        "Randomized parameters: maxBufferedDocs="
            + maxBufferedDocs
            + ", numSegments="
            + numSegments
            + ", docsPerSegment="
            + docsPerSegment
            + ", vectorProbability="
            + vectorProbability);

    GPUSearchParams params =
        new GPUSearchParams.Builder()
            .withCagraGraphBuildAlgo(cagraGraphBuildAlgo)
            .withIndexType(IndexType.BRUTE_FORCE)
            .build();

    // Configure with brute force index type
    CuVS2510GPUVectorsFormat bruteForceFormat =
        new CuVS2510GPUVectorsFormat(params); // Use brute force index

    IndexWriterConfig config =
        new IndexWriterConfig()
            .setCodec(alwaysKnnVectorsFormat(bruteForceFormat))
            .setMaxBufferedDocs(maxBufferedDocs)
            .setRAMBufferSizeMB(IndexWriterConfig.DISABLE_AUTO_FLUSH);

    int totalDocuments = numSegments * docsPerSegment;
    int totalExpectedVectors = 0;
    Set<Integer> docIDsHavingVectors = new HashSet<Integer>();

    try (IndexWriter writer = new IndexWriter(directory, config)) {
      for (int seg = 0; seg < numSegments; seg++) {
        int segmentVectorCount = 0;
        for (int i = 0; i < docsPerSegment; i++) {
          int docId = seg * docsPerSegment + i;
          Document doc = new Document();
          doc.add(new StringField(ID_FIELD, String.valueOf(docId), Field.Store.YES));

          // Randomly add vectors based on probability
          if (random.nextDouble() < vectorProbability) {
            float[] vector = dataProvider.getVectors(1)[0];
            doc.add(new KnnFloatVectorField(VECTOR_FIELD1, vector, EUCLIDEAN));
            docIDsHavingVectors.add(docId);
            segmentVectorCount++;
          }
          writer.addDocument(doc);
        }

        writer.commit();
        totalExpectedVectors += segmentVectorCount;

        log.log(
            Level.FINE,
            "Created brute force segment "
                + seg
                + ": "
                + docsPerSegment
                + " documents, "
                + segmentVectorCount
                + " with vectors");
      }

      log.log(
          Level.FINE,
          "Created "
              + numSegments
              + " brute force segments with "
              + totalDocuments
              + " total documents and "
              + totalExpectedVectors
              + " vectors");

      // Force merge all brute force segments
      writer.forceMerge(1);
      log.log(Level.FINE, "Forced merge of brute force segments completed");
    }

    // Verify the merged brute force index
    try (DirectoryReader reader = DirectoryReader.open(directory)) {
      List<LeafReaderContext> leaves = reader.leaves();
      assertEquals("Should have exactly one segment after merge", 1, leaves.size());
      LeafReader leafReader = leaves.get(0).reader();
      assertEquals("Total documents should match", totalDocuments, leafReader.maxDoc());

      // Count actual vectors in merged index
      var vectorValues = leafReader.getFloatVectorValues(VECTOR_FIELD1);
      int actualVectorCount = vectorValues != null ? vectorValues.size() : 0;

      log.log(
          Level.FINE,
          "Brute force merge results: Total documents: "
              + totalDocuments
              + ", Expected vectors: "
              + totalExpectedVectors
              + ", Actual vectors: "
              + actualVectorCount);

      assertEquals("Vector count should match expected", totalExpectedVectors, actualVectorCount);

      // Test brute force vector search (exact search)
      IndexSearcher searcher = new IndexSearcher(reader);
      float[] queryVector = dataProvider.getQueries(1)[0];
      int topK = dataProvider.getRandom(1, actualVectorCount);

      KnnFloatVectorQuery vectorQuery = new KnnFloatVectorQuery(VECTOR_FIELD1, queryVector, topK);
      TopDocs vectorResults = searcher.search(vectorQuery, topK);
      int numResults = vectorResults.scoreDocs.length;
      assertTrue("Should find some vector results in brute force index", numResults > 0);
      assertTrue("Should not find more vectors than exist", numResults <= actualVectorCount);

      log.log(
          Level.FINE,
          "Brute force search found "
              + numResults
              + " results out of "
              + actualVectorCount
              + " available vectors");

      assertEquals("Search should return exactly topK documents", topK, numResults);
      // Verify all returned documents are valid
      for (ScoreDoc scoreDoc : vectorResults.scoreDocs) {
        Document doc = searcher.storedFields().document(scoreDoc.doc);
        String docId = doc.get(ID_FIELD);
        assertNotNull("Document should have valid ID", docId);
        assertTrue("Score should be positive", scoreDoc.score > 0);
        assertTrue(
            "Document does not have a vector",
            docIDsHavingVectors.contains(Integer.parseInt(docId)));
      }
    }
  }

  /**
   * Test merging segments for {@link IndexType#CAGRA}
   * */
  @Test
  public void testMergeCagra() throws IOException {
    log.log(
        Level.FINE,
        "Starting testMergeCagraAndBruteForceIndex with CagraGraphBuildAlgo: "
            + cagraGraphBuildAlgo);

    // Use moderate dataset size
    int numSegments = dataProvider.getRandom(3, 10);
    int docsPerSegment = dataProvider.getRandom(20, 100);
    double vectorProbability = dataProvider.getRandom(0.2, 0.7);
    int maxBufferedDocs = dataProvider.getRandom(8, 17);

    log.log(
        Level.FINE,
        "Randomized parameters: maxBufferedDocs="
            + maxBufferedDocs
            + ", numSegments="
            + numSegments
            + ", docsPerSegment="
            + docsPerSegment
            + ", vectorProbability="
            + vectorProbability);

    // Configure with CAGRA + brute force combined index type
    GPUSearchParams params =
        new GPUSearchParams.Builder()
            .withCagraGraphBuildAlgo(cagraGraphBuildAlgo)
            .withIndexType(IndexType.CAGRA)
            .build();

    CuVS2510GPUVectorsFormat format = new CuVS2510GPUVectorsFormat(params);

    IndexWriterConfig config =
        new IndexWriterConfig()
            .setCodec(alwaysKnnVectorsFormat(format))
            .setMaxBufferedDocs(maxBufferedDocs)
            .setRAMBufferSizeMB(IndexWriterConfig.DISABLE_AUTO_FLUSH);

    int totalDocuments = numSegments * docsPerSegment;
    int totalExpectedVectors = 0;
    Set<Integer> docIDsHavingVectors = new HashSet<Integer>();

    try (IndexWriter writer = new IndexWriter(directory, config)) {
      for (int seg = 0; seg < numSegments; seg++) {
        int segmentVectorCount = 0;
        for (int i = 0; i < docsPerSegment; i++) {
          int docId = seg * docsPerSegment + i;
          Document doc = new Document();
          doc.add(new StringField(ID_FIELD, String.valueOf(docId), Field.Store.YES));

          // Randomly add vectors based on probability
          if (random.nextDouble() < vectorProbability) {
            float[] vector = dataProvider.getVectors(1)[0];
            doc.add(new KnnFloatVectorField(VECTOR_FIELD1, vector, EUCLIDEAN));
            docIDsHavingVectors.add(docId);
            segmentVectorCount++;
          }
          writer.addDocument(doc);
        }

        writer.commit();
        totalExpectedVectors += segmentVectorCount;

        log.log(
            Level.FINE,
            "Created CAGRA segment "
                + seg
                + ": "
                + docsPerSegment
                + " documents, "
                + segmentVectorCount
                + " with vectors");
      }

      log.log(
          Level.FINE,
          "Created "
              + numSegments
              + " CAGRA segments with "
              + totalDocuments
              + " total documents and "
              + totalExpectedVectors
              + " vectors");

      // Force merge all CAGRA segments
      writer.forceMerge(1);
      log.log(Level.FINE, "Forced merge of brute force segments completed");
    }

    // Verify the merged CAGRA index
    try (DirectoryReader reader = DirectoryReader.open(directory)) {
      List<LeafReaderContext> leaves = reader.leaves();
      assertEquals("Should have exactly one segment after merge", 1, leaves.size());
      LeafReader leafReader = leaves.get(0).reader();
      assertEquals("Total documents should match", totalDocuments, leafReader.maxDoc());

      // Count actual vectors in merged index
      var vectorValues = leafReader.getFloatVectorValues(VECTOR_FIELD1);
      int actualVectorCount = vectorValues != null ? vectorValues.size() : 0;

      log.log(
          Level.FINE,
          "CAGRA merge results: Total documents: "
              + totalDocuments
              + ", Expected vectors: "
              + totalExpectedVectors
              + ", Actual vectors: "
              + actualVectorCount);

      assertEquals("Vector count should match expected", totalExpectedVectors, actualVectorCount);

      // Test CAGRA vector search (exact search)
      IndexSearcher searcher = new IndexSearcher(reader);
      float[] queryVector = dataProvider.getQueries(1)[0];
      int topK = dataProvider.getRandom(1, actualVectorCount);

      KnnFloatVectorQuery vectorQuery = new KnnFloatVectorQuery(VECTOR_FIELD1, queryVector, topK);
      TopDocs vectorResults = searcher.search(vectorQuery, topK);
      int numResults = vectorResults.scoreDocs.length;
      assertTrue("Should find some vector results in CAGRA + brute force index", numResults > 0);
      assertTrue("Should not find more vectors than exist", numResults <= actualVectorCount);

      log.log(
          Level.FINE,
          "CAGRA search found "
              + numResults
              + " results out of "
              + actualVectorCount
              + " available vectors");

      assertEquals("Search should return exactly topK documents", topK, numResults);
      // Verify all returned documents are valid
      for (ScoreDoc scoreDoc : vectorResults.scoreDocs) {
        Document doc = searcher.storedFields().document(scoreDoc.doc);
        String docId = doc.get(ID_FIELD);
        assertNotNull("Document should have valid ID", docId);
        assertTrue("Score should be positive", scoreDoc.score > 0);
        assertTrue(
            "Document does not have a vector",
            docIDsHavingVectors.contains(Integer.parseInt(docId)));
      }
    }
  }

  /**
   * Test merging segments for {@link IndexType#CAGRA_AND_BRUTE_FORCE}
   * */
  @Test
  public void testMergeCagraAndBruteForceIndex() throws IOException {
    int numSegments = dataProvider.getRandom(3, 10);
    int docsPerSegment = dataProvider.getRandom(20, 100);
    double vectorProbability = dataProvider.getRandom(0.2, 0.7);
    int maxBufferedDocs = dataProvider.getRandom(8, 17);

    log.log(
        Level.FINE,
        "Randomized parameters: maxBufferedDocs="
            + maxBufferedDocs
            + ", numSegments="
            + numSegments
            + ", docsPerSegment="
            + docsPerSegment
            + ", vectorProbability="
            + vectorProbability);

    GPUSearchParams params =
        new GPUSearchParams.Builder()
            .withCagraGraphBuildAlgo(cagraGraphBuildAlgo)
            .withIndexType(IndexType.CAGRA_AND_BRUTE_FORCE)
            .build();

    CuVS2510GPUVectorsFormat format = new CuVS2510GPUVectorsFormat(params);

    IndexWriterConfig config =
        new IndexWriterConfig()
            .setCodec(alwaysKnnVectorsFormat(format))
            .setMaxBufferedDocs(maxBufferedDocs)
            .setRAMBufferSizeMB(IndexWriterConfig.DISABLE_AUTO_FLUSH);

    int totalDocuments = numSegments * docsPerSegment;
    int totalExpectedVectors = 0;
    Set<Integer> docIDsHavingVectors = new HashSet<Integer>();

    try (IndexWriter writer = new IndexWriter(directory, config)) {
      for (int seg = 0; seg < numSegments; seg++) {
        int segmentVectorCount = 0;
        for (int i = 0; i < docsPerSegment; i++) {
          int docId = seg * docsPerSegment + i;
          Document doc = new Document();
          doc.add(new StringField(ID_FIELD, String.valueOf(docId), Field.Store.YES));

          // Randomly add vectors based on probability
          if (random.nextDouble() < vectorProbability) {
            float[] vector = dataProvider.getVectors(1)[0];
            doc.add(new KnnFloatVectorField(VECTOR_FIELD1, vector, EUCLIDEAN));
            docIDsHavingVectors.add(docId);
            segmentVectorCount++;
          }
          writer.addDocument(doc);
        }

        writer.commit();
        totalExpectedVectors += segmentVectorCount;

        log.log(
            Level.FINE,
            "Created CAGRA + brute force segment "
                + seg
                + ": "
                + docsPerSegment
                + " documents, "
                + segmentVectorCount
                + " with vectors");
      }

      log.log(
          Level.FINE,
          "Created "
              + numSegments
              + " CAGRA + brute force segments with "
              + totalDocuments
              + " total documents and "
              + totalExpectedVectors
              + " vectors");

      // Force merge all CAGRA + brute force segments
      writer.forceMerge(1);
      log.log(Level.FINE, "Forced merge of brute force segments completed");
    }

    // Verify the merged CAGRA + brute force index
    try (DirectoryReader reader = DirectoryReader.open(directory)) {
      List<LeafReaderContext> leaves = reader.leaves();
      assertEquals("Should have exactly one segment after merge", 1, leaves.size());
      LeafReader leafReader = leaves.get(0).reader();
      assertEquals("Total documents should match", totalDocuments, leafReader.maxDoc());

      // Count actual vectors in merged index
      var vectorValues = leafReader.getFloatVectorValues(VECTOR_FIELD1);
      int actualVectorCount = vectorValues != null ? vectorValues.size() : 0;

      log.log(
          Level.FINE,
          "CAGRA + brute force merge results: Total documents: "
              + totalDocuments
              + ", Expected vectors: "
              + totalExpectedVectors
              + ", Actual vectors: "
              + actualVectorCount);

      assertEquals("Vector count should match expected", totalExpectedVectors, actualVectorCount);

      // Test CAGRA + brute force vector search (exact search)
      IndexSearcher searcher = new IndexSearcher(reader);
      float[] queryVector = dataProvider.getQueries(1)[0];
      int topK = dataProvider.getRandom(1, actualVectorCount);

      KnnFloatVectorQuery vectorQuery = new KnnFloatVectorQuery(VECTOR_FIELD1, queryVector, topK);
      TopDocs vectorResults = searcher.search(vectorQuery, topK);
      int numResults = vectorResults.scoreDocs.length;
      assertTrue("Should find some vector results in CAGRA + brute force index", numResults > 0);
      assertTrue("Should not find more vectors than exist", numResults <= actualVectorCount);

      log.log(
          Level.FINE,
          "CAGRA + Brute force search found "
              + numResults
              + " results out of "
              + actualVectorCount
              + " available vectors");

      assertEquals("Search should return exactly topK documents", topK, numResults);
      // Verify all returned documents are valid
      for (ScoreDoc scoreDoc : vectorResults.scoreDocs) {
        Document doc = searcher.storedFields().document(scoreDoc.doc);
        String docId = doc.get(ID_FIELD);
        assertNotNull("Document should have valid ID", docId);
        assertTrue("Score should be positive", scoreDoc.score > 0);
        assertTrue(
            "Document does not have a vector",
            docIDsHavingVectors.contains(Integer.parseInt(docId)));
      }
    }
  }
}
