
object Cells {
  /* ... new cell ... */

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

  val loadPath = "/home/henry/data/yelp/output/featuresDF_CA.parquet"
  val inputDF = spark.read.parquet(loadPath)

  /* ... new cell ... */

  val label: Array[Double] = inputDF.select("label").rdd.map{
    case Row(value: Double) => value
  }.collect()

  /* ... new cell ... */

  import java.io._
  val lable_file = "/home/henry/data/yelp/output/labels.csv"
  val label_writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(lable_file)))
  for (x <- label) {
    label_writer.write(x + "\n") 
  }
  label_writer.close()

  /* ... new cell ... */

  val features = inputDF.select("features").rdd.map{case Row(value: Vector) => value.toArray}.collect()

  /* ... new cell ... */

  val feature_file = "/home/henry/data/yelp/output/features.csv"
  val feature_writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(feature_file)))
  for (f <- features) {
    val len = f.length-1;
    for(i <- 0 until len){
      feature_writer.write(f(i) + ",") 
    }
    feature_writer.write(f(len) + "\n")
  }
  feature_writer.close()

  /* ... new cell ... */
}
                  