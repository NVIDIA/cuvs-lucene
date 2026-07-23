/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static org.apache.lucene.tests.util.LuceneTestCase.newIndexWriterConfig;
import static org.apache.lucene.tests.util.LuceneTestCase.newTieredMergePolicy;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.TreeMap;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.apache.lucene.codecs.Codec;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.store.Directory;
import org.apache.lucene.tests.analysis.MockAnalyzer;
import org.apache.lucene.tests.analysis.MockTokenizer;
import org.apache.lucene.tests.index.RandomIndexWriter;

public class TestUtils {

  private static final Logger log = Logger.getLogger(TestUtils.class.getName());

  public static float[][] generateRandomVectors(Random random, int size, int dimensions) {
    float[][] dataset = new float[size][dimensions];
    for (int i = 0; i < size; i++) {
      for (int j = 0; j < dimensions; j++) {
        dataset[i][j] = random.nextFloat() * 100;
      }
    }
    return dataset;
  }

  public static List<List<Integer>> generateExpectedTopK(
      int topK, float[][] dataset, float[][] queries) {
    List<List<Integer>> neighborsResult = new ArrayList<>();
    int dimensions = dataset[0].length;

    for (float[] query : queries) {
      Map<Integer, Double> distances = new TreeMap<>();
      for (int j = 0; j < dataset.length; j++) {
        double distance = 0;
        for (int k = 0; k < dimensions; k++) {
          distance += (query[k] - dataset[j][k]) * (query[k] - dataset[j][k]);
        }
        distances.put(j, (distance));
      }

      Map<Integer, Double> sorted = new TreeMap<Integer, Double>(distances);
      log.log(Level.FINER, "EXPECTED: " + sorted);

      // Sort by distance and select the topK nearest neighbors
      List<Integer> neighbors =
          distances.entrySet().stream()
              .sorted(Map.Entry.comparingByValue())
              .map(Map.Entry::getKey)
              .toList();
      neighborsResult.add(neighbors.subList(0, Math.min(topK * 3, dataset.length)));
    }

    log.log(Level.FINE, "Expected results generated successfully.");
    return neighborsResult;
  }

  public static RandomIndexWriter createWriter(Random random, Directory directory, Codec codec)
      throws IOException {
    return new RandomIndexWriter(
        random,
        directory,
        newIndexWriterConfig(new MockAnalyzer(random, MockTokenizer.SIMPLE, true))
            .setCodec(codec)
            .setMergePolicy(newTieredMergePolicy()));
  }

  public static IndexWriterConfig createWriterConfig(Random random, Codec codec) {
    return newIndexWriterConfig(new MockAnalyzer(random, MockTokenizer.SIMPLE, true))
        .setCodec(codec)
        .setMergePolicy(newTieredMergePolicy());
  }

  public static String generateRandomText(Random random, int length) {
    StringBuilder sb = new StringBuilder(length);
    String chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    for (int i = 0; i < length; i++) {
      sb.append(chars.charAt(random.nextInt(chars.length())));
    }
    return sb.toString();
  }
}
