import java.io.*;
import java.util.*;

public class CsvParser {
    // features: N = 37680, D = 59;
    // labels: N = 37680, D = 1;
    public static double[][] parse(String fileName, int N, int D) {
        File file= new File(fileName);
        double[][] matrix = new double[N][D];
        Scanner inputStream;
        try{
            inputStream = new Scanner(file);
            int i=0;
            while(inputStream.hasNext()){
                String line= inputStream.next();
                String[] values = line.split(",");
                for(int j=0; j<values.length; j++){
                    matrix[i][j] = Double.valueOf(values[j]);
                }
                i++;
            }
            inputStream.close();
        }catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        return matrix;
    }
        
}