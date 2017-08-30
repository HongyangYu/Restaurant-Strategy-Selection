
object Cells {
  :cp /home/henry/Software/spark-notebook/spark-notebook-0.7.0-scala-2.11.8-spark-2.1.0-hadoop-2.7.2/external_lib/opennlp-tools-1.7.2.jar

  /* ... new cell ... */

  import opennlp.tools.cmdline.parser.ParserTool
  import opennlp.tools.parser.Parse
  import opennlp.tools.parser.ParserFactory
  import opennlp.tools.parser.ParserModel
  import java.io.FileInputStream

  /* ... new cell ... */

  import org.apache.spark.sql.SparkSession
  
  val spark = SparkSession
    .builder()
    .appName("yelp")
    .config("spark.some.config.option", "some-value")
    .getOrCreate()
  
  // For implicit conversions like converting RDDs to DataFrames
  import spark.implicits._
  val dfBusiness = spark.read.json("/home/henry/data/yelp/yelp_academic_dataset_business.json")
  
  //df.groupBy("state")
    //.count()
    //.sort(desc("count"))
    //.first()
  //result:"AZ"
  var azBusiness = dfBusiness.select("business_id")
    .filter($"state" === "PA").as[String].collect()
  //df2.filter(not($"c4".isin(c2List: _*)))
  //dfBusiness.filter($"business_id".isin(azBusiness: _*)).take(10)
  
  val azReview = spark.read.json("/home/henry/data/yelp/yelp_academic_dataset_tip.json")
    .filter($"business_id".isin(azBusiness: _*))
  val stringReview = azReview.select("text") //.limit(100)
    .as[String]
    .collect()
    .mkString(" ")

  /* ... new cell ... */

  var nounPhrases = ""
  def getNounPhrases(p: Parse) {
    if (p.getType == "NN") {
      nounPhrases += p.getCoveredText + " "
    }
    for (child <- p.getChildren) {
      getNounPhrases(child)
    }
  }

  /* ... new cell ... */

  val is = new FileInputStream("/home/henry/data/yelp/en-parser-chunking.bin")
  val parserModel = new ParserModel(is)
  val parser = ParserFactory.create(parserModel)
  val topParses = ParserTool.parseLine(stringReview, parser, 1)
  for (p <- topParses) {
    getNounPhrases(p)
  }
  is.close()

  /* ... new cell ... */

  import java.io._
  val pw = new PrintWriter(new File("/home/henry/data/yelp/output/nounPhrases.txt" ))
  pw.write(nounPhrases)
  pw.close

  /* ... new cell ... */

  import opennlp.tools.tokenize.TokenizerModel
  import opennlp.tools.tokenize.TokenizerME
  import java.io.FileInputStream
  val tokenIs = new FileInputStream("/home/henry/data/yelp/en-token.bin");
  val tokenModel = new TokenizerModel(tokenIs);
  val tokenizer = new TokenizerME(tokenModel);
  val rddToken = sc.parallelize(nounPhrases.split(" "))
  val allFeature = rddToken.map(word=>(word,1))
    .reduceByKey((a,b)=>(a+b))
    .sortBy(entry => -entry._2)
    .toDF()
    .select("_1")
    .filter(_!=" ")
    .filter($"_1".isNotNull)
    .as[String]
    .collect()
    .toList
  tokenIs.close()

  /* ... new cell ... */

  val textDF = azReview.withColumn("content", explode(split($"text".as[String], " "))).select($"business_id", $"content") //dataframe
  
  val finalRDD = textDF.filter(textDF("content").isin(allFeature:_*)).rdd.map(x=>(x.get(0).toString, x.get(1).toString)).groupByKey().mapValues(_.toList)
  
  var finalDS = finalRDD.toDS()
  
  import org.apache.spark.ml.feature.Word2Vec
  import org.apache.spark.ml.linalg.Vector
  import org.apache.spark.sql.Row
  val word2Vec = new Word2Vec()
    .setInputCol("_2")
    .setOutputCol("vector")
    .setVectorSize(30)
    .setMinCount(0)
  val fitModel = word2Vec.fit(finalDS)
  val id_vecDF = fitModel.transform(finalDS).drop("_2")
  
  //id_vecDF is the final result

  /* ... new cell ... */

  val savePath = "/home/henry/data/yelp/output/feature2_PA.parquet"
  id_vecDF.write.parquet(savePath)
}
                  