import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.*;

public class Yelp_DL4J_Regression {
	// Number of iterations per minibatch
	public static final int iterations = 10;
	// Number of epochs (full passes of the data)
	public static final int nEpochs = 200;
	// Batch size: i.e., each epoch has nSamples/batchSize parameter updates
	public static final int batchSize = 100;
	// Network learning rate
	public static final double learningRate = 0.01;

	// Random number generator seed, for reproducability
	private static final int seed = 12345;
	private static final Random rand = new Random(seed);
	
	// feature number and dimension
	public static final int featureDim = 59;
	public static final int trainNum = 37000;
	public static final int testNum = 680;
	
	// train data file
	public static final String featurePath_train = "/home/henry/data/yelp/output/features_train.csv";
	public static final String labelPath_train = "/home/henry/data/yelp/output/labels_train.csv";
	// test data file
	public static final String featurePath_test = "/home/henry/data/yelp/output/features_test.csv";
	public static final String labelPath_test = "/home/henry/data/yelp/output/labels_test.csv";
	// model path
	public static final String modelPath = "YelpMultiLayerNetwork.zip";
	
	public static void regressionTrain() {
		// Generate the training data
		DataSetIterator iterator = getData(featurePath_train, labelPath_train, trainNum);

		// Create the network
		int numInput = featureDim;
		int numOutputs = 1;
		int nHidden = 10;
		MultiLayerNetwork net = new MultiLayerNetwork(new NeuralNetConfiguration.Builder().seed(seed)
				.iterations(iterations).optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.learningRate(learningRate).weightInit(WeightInit.XAVIER).updater(Updater.NESTEROVS).momentum(0.9)
				.list()
				.layer(0, new DenseLayer.Builder().nIn(numInput).nOut(nHidden).activation(Activation.TANH).build())
				.layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MSE).activation(Activation.IDENTITY)
						.nIn(nHidden).nOut(numOutputs).build())
				.pretrain(false).backprop(true).build());
		net.init();
		net.setListeners(new ScoreIterationListener(1));

		// Train the network on the full data set, and evaluate in periodically
		for (int i = 0; i < nEpochs; i++) {
			iterator.reset();
			net.fit(iterator);
		}

		// Save the model
		File locationToSave = new File(modelPath); 
		boolean saveUpdater = true; 
		try {
			ModelSerializer.writeModel(net, locationToSave, saveUpdater);
		} catch (IOException e) {
			e.printStackTrace();
		}

		System.out.println("Training end successfully.");
	}
	
	public static void regressionValidate(){
		//Load the model
        MultiLayerNetwork net;
		try {
			net = ModelSerializer.restoreMultiLayerNetwork(modelPath);
			DataSetIterator iteratorTestData = getData(featurePath_test, labelPath_test, testNum);
			// Test the network on the full data set, and evaluate in periodically
			INDArray out = net.output(iteratorTestData, false);
			System.out.println("\n\nLength: " + out.length());
//			System.out.println(out);
			
			save("result.csv", out);
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		System.out.println("Validation end successfully.");
	}
	
	
	
	private static void save(String fileName, INDArray out){
		try {
    	    PrintWriter pr = new PrintWriter(fileName);  
    	    for (int i=0; i<out.length(); i++) {
    	        pr.println(out.getDouble(i));
    	    }
    	    pr.close();
    	}
    	catch (Exception e) {
    	    e.printStackTrace();
    	    System.out.println("No such file exists.");
    	}
	}

	private static DataSetIterator getData(String featurePath, String labelPath, int N) {
		double[][] feature = CsvParser.parse(featurePath, N, featureDim);
		double[][] label = CsvParser.parse(labelPath, N, 1);

		INDArray inputNDArray = Nd4j.create(feature);
		INDArray outputNDArray = Nd4j.create(label);
		DataSet dataSet = new DataSet(inputNDArray, outputNDArray);
		List<DataSet> listDS = dataSet.asList();
		Collections.shuffle(listDS, rand);

		return new ListDataSetIterator(listDS, batchSize);
	}

}