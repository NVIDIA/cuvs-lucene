/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuvs.lucene;

import static com.nvidia.cuvs.lucene.TestUtils.generateRandomVectors;

import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;

public class TestDataProvider {

  private static final Logger log = Logger.getLogger(TestDataProvider.class.getName());
  private static final int DATASET_SIZE_LIMIT = 1000;
  private static final int DATASET_SIZE_MIN = 200;
  private static final int DIMENSIONS_LIMIT = 256;
  private static final int DIMENSIONS_MIN = 8;
  private static final int TOP_K_LIMIT = 64;
  private static final int TOP_K_MIN = 2;
  private static final int QUERIES_LIMIT = 50;
  private static final int QUERIES_MIN = 2;

  public static final String ID_FIELD = "id";
  public static final String TEXT_FIELD = "some_text_field";
  public static final String CATEGORY_FIELD = "category_field";
  public static final String VECTOR_FIELD1 = "vector_field1";
  public static final String VECTOR_FIELD2 = "vector_field2";

  private int datasetSize;
  private int dimensions;
  private int topK;
  private float[][] dataset1;
  private float[][] dataset2;
  private int numQueries;
  private Random random;

  public TestDataProvider(Random random) {
    datasetSize = random.nextInt(DATASET_SIZE_MIN, DATASET_SIZE_LIMIT);
    dimensions = random.nextInt(DIMENSIONS_MIN, DIMENSIONS_LIMIT);
    topK = Math.min(random.nextInt(TOP_K_MIN, TOP_K_LIMIT), datasetSize);
    dataset1 = generateRandomVectors(random, datasetSize, dimensions);
    dataset2 = generateRandomVectors(random, datasetSize, dimensions);
    numQueries = random.nextInt(QUERIES_MIN, QUERIES_LIMIT);
    this.random = random;
    log.log(Level.FINE, "Dataset size: " + datasetSize + "x" + dimensions + ", topK: " + topK);
  }

  public int getDatasetSize() {
    return datasetSize;
  }

  public int getDimensions() {
    return dimensions;
  }

  public int getTopK() {
    return topK;
  }

  public float[][] getDataset1() {
    return dataset1;
  }

  public float[][] getDataset2() {
    return dataset2;
  }

  public float[][] getQueries(int numQueries) {
    return generateRandomVectors(random, numQueries, dimensions);
  }

  public float[][] getVectors(int numVectors) {
    return generateRandomVectors(random, numVectors, dimensions);
  }

  public int getRandom(int min, int max) {
    return random.nextInt(min, max);
  }

  public double getRandom(double min, double max) {
    return random.nextDouble(min, max);
  }

  public int getNumQueries() {
    return numQueries;
  }
}
