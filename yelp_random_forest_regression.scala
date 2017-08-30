
object Cells {
  import org.apache.spark.ml.linalg._
  import org.apache.spark.sql._
  import org.apache.spark.sql.functions._
  import scala.collection.mutable.WrappedArray
  import org.apache.spark.ml.feature._
  
  val spark = SparkSession
    .builder()
    .appName("Spark Yelp")
    .config("spark.some.config.option", "some-value")
    .getOrCreate()
  
  // For implicit conversions like converting RDDs to DataFrames
  import spark.implicits._

  /* ... new cell ... */

  import org.apache.spark.ml.Pipeline
  import org.apache.spark.ml.evaluation.RegressionEvaluator
  import org.apache.spark.ml.regression._

  /* ... new cell ... */

  val loadPath = "/home/henry/data/yelp/output/featuresDF.parquet"
  val inputDF = spark.read.parquet(loadPath)

  /* ... new cell ... */

  // Automatically identify categorical features, and index them.
  // Set maxCategories so features with > 4 distinct values are treated as continuous.
  val featureIndexer = new VectorIndexer()
    .setInputCol("features")
    .setOutputCol("indexedFeatures")
    .setMaxCategories(40)
    .fit(inputDF)

  /* ... new cell ... */

  // Split the data into training and test sets (30% held out for testing)
  val Array(trainData, testData) = inputDF.randomSplit(Array(0.7, 0.3))

  /* ... new cell ... */

  // Train a RandomForest model.
  val rf = new RandomForestRegressor()
    .setLabelCol("label")
    .setFeaturesCol("features")
  
  // Chain indexer and forest in a Pipeline
  val pipeline = new Pipeline()
    .setStages(Array(featureIndexer, rf))
  
  // Train model.  This also runs the indexer.
  val model = pipeline.fit(trainData)
  
  // Make predictions.
  val predictions = model.transform(testData)

  /* ... new cell ... */

  // Select example rows to display.
  predictions.select("prediction", "label", "features")

  /* ... new cell ... */

  val result = predictions.select("prediction", "label")

  /* ... new cell ... */

  result
     // place all data in a single partition 
     .coalesce(1)
     .write.format("com.databricks.spark.csv")
     .option("header", "true")
     .save("/home/henry/data/yelp/output/result_yelp_randomForest.csv")

  /* ... new cell ... */

  // Select (prediction, true label) and compute test error
  val evaluator = new RegressionEvaluator()
    .setLabelCol("label")
    .setPredictionCol("prediction")
    .setMetricName("rmse")
  val rmse = evaluator.evaluate(predictions)
  println("Root Mean Squared Error (RMSE) on test data = " + rmse)

  /* ... new cell ... */

  val rfModel = model.stages(1).asInstanceOf[RandomForestRegressionModel]
  println("Learned regression forest model:\n" + rfModel.toDebugString)
}
                  