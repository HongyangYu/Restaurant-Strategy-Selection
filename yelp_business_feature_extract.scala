
object Cells {
  import org.apache.spark.ml.feature.Word2Vec
  import org.apache.spark.ml.linalg.Vector
  import org.apache.spark.sql.Row
  import org.apache.spark.sql.functions._
  import org.apache.spark.sql.SparkSession
  import org.apache.spark.sql.SQLImplicits
  import scala.collection.mutable.WrappedArray
  import org.apache.spark.ml.feature._
  
  val spark = SparkSession
    .builder()
    .appName("Spark SQL basic example")
    .config("spark.some.config.option", "some-value")
    .getOrCreate()
  
  // For implicit conversions like converting RDDs to DataFrames
  import spark.implicits._

  /* ... new cell ... */

  val path = "/home/henry/data/yelp/yelp_academic_dataset_business.json"
  val inputDF = sparkSession.read.json(path)

  /* ... new cell ... */

  val businessDF = inputDF.select($"business_id", $"categories", $"attributes", $"stars").filter(
    $"categories".isNotNull and $"attributes".isNotNull and $"state" === "PA")

  /* ... new cell ... */

  val categories_feature_size = 20

  /* ... new cell ... */

  val word2Vec = new Word2Vec()
    .setInputCol("categories")
    .setOutputCol("categories_feature")
    .setVectorSize(categories_feature_size)
    .setMinCount(0)

  /* ... new cell ... */

  val model = word2Vec.fit(businessDF)

  /* ... new cell ... */

  val categoryFeatureDF = model.transform(businessDF).drop("categories")

  /* ... new cell ... */

  val attrDF = categoryFeatureDF.withColumn("attributes", explode($"attributes"))
  
  val allAttr = attrDF.select(substring_index(col("attributes"), ":", 1).as("attri")).distinct
  val attrIndexer = new StringIndexer()
    .setInputCol("attri")
    .setOutputCol("attrIndex")
    .fit(allAttr)
  val attriIdx = attrIndexer.transform(allAttr)

  /* ... new cell ... */

  val split_atrr = attrDF.withColumn("attriVal", split($"attributes", ": ")).select(
    $"business_id", $"categories_feature", $"attriVal".getItem(0).as("attri"), $"attriVal".cast("string"), $"stars")
  
  val attrValDF = split_atrr.withColumn("attrInt", when($"attriVal".contains("True"), 1.toInt).when(
    $"attriVal".contains("False") or $"attriVal".contains("none"), 0.toInt).when(
    $"attriVal".contains("RestaurantsPriceRange"), $"attriVal".substr(length($"attriVal")-1, length($"attriVal")-length($"attriVal")+1).cast("int"))
                                           .otherwise(1)).drop("attriVal")

  /* ... new cell ... */

  val attributeDF = attrValDF.join(attriIdx, attrValDF("attri")===attriIdx("attri")).select(
    $"business_id", $"categories_feature", $"attrIndex",$"attrInt", $"stars")

  /* ... new cell ... */

  // Create a sorted array of attributes
  val attributesArray = attributeDF
    .select($"attrIndex")
    .distinct.map(_.getDouble(0).toInt.toString)
    .collect
    .sorted

  /* ... new cell ... */

  // Prepare vector assemble
  val assembler =  new VectorAssembler()
    .setInputCols(attributesArray)
    .setOutputCol("attri_feature")
  // Aggregation expressions
  val exprs = attributesArray.map(
     c => sum(when($"attrIndex" === c, $"attrInt").otherwise(lit(0))).alias(c))

  /* ... new cell ... */

  val cate_attr_featuresDF = assembler.transform(
      attributeDF.groupBy($"business_id", $"categories_feature", $"stars").agg(exprs.head, exprs.tail: _*))
    .select($"business_id", $"categories_feature", $"attri_feature", $"stars")

  /* ... new cell ... */

  val cate_attr_featuresDF2 =assembler.transform(
      attributeDF.groupBy($"business_id", $"categories_feature", $"stars").agg(exprs.head, exprs.tail: _*)).drop($"attri_feature")

  /* ... new cell ... */

  val attr_featuresDF2 = cate_attr_featuresDF2.drop("categories_feature")

  /* ... new cell ... */

  

  /* ... new cell ... */

  val assemblerFeatures = new VectorAssembler()
    .setInputCols(Array("categories_feature", "attri_feature"))
    .setOutputCol("features")
  
  val mergeFeatureDF = assemblerFeatures.transform(cate_attr_featuresDF).drop($"categories_feature").drop($"attri_feature")

  /* ... new cell ... */

  val denseFunc = (vect: Vector) => Vectors.dense(vect.toArray)
  val denseUDF = udf(denseFunc)

  /* ... new cell ... */

  val mergeFeatureRDD = mergeFeatureDF.as[(String, Double, Vector)].rdd

  /* ... new cell ... */

  import org.apache.spark.ml.linalg.Vectors
  import org.apache.spark.ml.feature.LabeledPoint
  
  val labeledRDD = mergeFeatureRDD.map(
    line => LabeledPoint(line._2, Vectors.dense(line._3.toArray))
  )

  /* ... new cell ... */

  val labeledDF = labeledRDD.toDF
  
  /* ... new cell ... */

  labeledDF.count

  /* ... new cell ... */

  val savePath = "/home/henry/data/yelp/output/featuresDF_PA.parquet"
  labeledDF.write.parquet(savePath)
}
                  