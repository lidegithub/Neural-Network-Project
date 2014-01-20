/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package timeseriesforecast;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintStream;
import java.util.Vector;
import java.util.logging.Level;
import java.util.logging.Logger;
import rcaller.RCaller;
import rcaller.RCode;

/**
 *
 * @author Ega
 */
public class Comparison {

    private int numOfInputUnit;
    private int numOfHiddenUnit;
    private int numOfOutputUnit;
    private double[][] V_FFNN;
    private double[][] W_FFNN;
    private double[][] V_RNN;
    private double[][] W_RNN;
    private double[][] U_RNN;
    private double[][] V_FFNNH;
    private double[][] W_FFNNH;
    private double[][] V_RNNH;
    private double[][] W_RNNH;
    private double[][] U_RNNH;
    private double eta;
    private double alpha;
    private int maxEpoch;
    private double maxError;
    private int weightInitialization;

    public Comparison(int weightInitialization, int numOfInputUnit, int numOfHiddenUnit, int numOfOutputUnit, double eta, double alpha, int maxEpoch, double maxError) {
        this.numOfInputUnit = numOfInputUnit;
        this.numOfHiddenUnit = numOfHiddenUnit;
        this.numOfOutputUnit = numOfOutputUnit;
        this.eta = eta;
        this.alpha = alpha;
        this.maxEpoch = maxEpoch;
        this.maxError = maxError;
        this.weightInitialization = weightInitialization;
    }

    public Vector getDataSetNYB() {
        Vector results = new Vector();
        double[] trainingSet;
        double[] testingSet;
        double[] validationSet;
        try {
            /* Creating a RCaller */
            RCaller caller = new RCaller();
            caller.setRscriptExecutable("/usr/bin/Rscript");

            /* Creating a source code */
            RCode code = new RCode();
            code.clear();

            // add libraries needed to load data set
            code.addRCode("library(zoo)");
            code.addRCode("library(timeSeries)");
            code.addRCode("library(rdatamarket)");

            //get data for training, testing and validation set
            code.addRCode("dminit(\"be0123ef782e49348a7ed53c2444c08c\")");
            code.addRCode("dataSet <- dmlist(\"22nv\")");
            code.addRCode("trainingSet <- dataSet[1:96,2]");
            code.addRCode("testingSet <- dataSet[97:132,2]");
            code.addRCode("validationSet <- dataSet[133:168,2]");
            code.addRCode("results<-list(trainSet = trainingSet, testSet = testingSet, validSet = validationSet, data = dataSet[,2])");
            caller.setRCode(code);
            System.out.println("script exe" + caller.getRscriptExecutable());
            caller.runAndReturnResult("results");
            trainingSet = caller.getParser().getAsDoubleArray("trainSet");
            testingSet = caller.getParser().getAsDoubleArray("testSet");
            validationSet = caller.getParser().getAsDoubleArray("validSet");
            double[] allData = caller.getParser().getAsDoubleArray("data");
            results.addElement(trainingSet);
            results.addElement(testingSet);
            results.addElement(validationSet);
            results.addElement(allData);
        } catch (Exception ex) {
            System.out.println(ex.getMessage());
        }
        return results;
    }

    public double [] HybridFFNN(double[][] V, double[][] W) {
        TimeSeriesHybridFFNN tsnn = new TimeSeriesHybridFFNN(V, W);
        Vector dataSet = getDataSetNYB();
        tsnn.setMinMax((double[]) dataSet.elementAt(3));

        double[] trainingSet = tsnn.normalizeData((double[]) dataSet.elementAt(0));
        double[] testingSet = tsnn.normalizeData((double[]) dataSet.elementAt(1));
        double[] validationSet = tsnn.normalizeData((double[]) dataSet.elementAt(2));

        double[] RMSE = tsnn.TrainingNN(weightInitialization, trainingSet, testingSet, numOfInputUnit, numOfHiddenUnit, numOfOutputUnit, eta, alpha, maxEpoch, maxError);
        try {
         RCaller caller = new RCaller();
         caller.setRscriptExecutable("/usr/bin/Rscript");
         caller.cleanRCode();
         File file;
         String[] arr = new String[1];
         arr[0] = "New York Birth";
         file = caller.startPlot();
         caller.addDoubleArray("RMSE.hybrid.ffnn", RMSE);
         caller.addStringArray("arr", arr);
         caller.addRCode("a = arr[1]");
         caller.addRCode("plot.ts(RMSE.hybrid.ffnn, main=a)");
         caller.endPlot();
         caller.runOnly();
         caller.showPlot(file);
         } catch (IOException ex) {
         Logger.getLogger(FFNN.class.getName()).log(Level.SEVERE, null, ex);
         }
        

        Vector forecastError = tsnn.generalizationNN(validationSet,numOfInputUnit, numOfHiddenUnit, numOfOutputUnit,eta, alpha);
        double [] SE = (double[]) forecastError.elementAt(0);
        double [] absE = (double[]) forecastError.elementAt(1);
        double [] absPE = (double[]) forecastError.elementAt(2);
        double RMSEValidation = tsnn.calculateForecastError(SE);
        double MAE = tsnn.calculateMAE(absE);
        double MAPE = tsnn.calculateMAPE(absPE);      
        //System.out.println("Forecast RMSE = "+RMSEValidation);
        double [] result = new double[3];
        result[0] = RMSEValidation;
        result[1] = MAE;
        result[2] = MAPE;
        return result;
    }

    public double [] RNN(double[][] V, double[][] W, double[][] U) {
        TimeSeriesRNN tsnn = new TimeSeriesRNN(V, W, U);
        Vector dataSet = getDataSetNYB();
        tsnn.setMinMax((double[]) dataSet.elementAt(3));       
        double[] trainingSet = tsnn.normalizeData((double[]) dataSet.elementAt(0));
        double[] testingSet = tsnn.normalizeData((double[]) dataSet.elementAt(1));
        double[] validationSet = tsnn.normalizeData((double[]) dataSet.elementAt(2));
        double[] RMSE = tsnn.TrainingNN(weightInitialization, trainingSet, testingSet, numOfInputUnit, numOfHiddenUnit, numOfOutputUnit, eta, alpha, maxEpoch, maxError);
        
         try {
            RCaller caller = new RCaller();
            caller.setRscriptExecutable("/usr/bin/Rscript");
            caller.cleanRCode();
            File file;
            String[] arr = new String[1];
            arr[0] = "New York Birth";
            file = caller.startPlot();
            caller.addDoubleArray("RMSE.rnn", RMSE);
            caller.addStringArray("arr", arr);
            caller.addRCode("a = arr[1]");
            caller.addRCode("plot.ts(RMSE.rnn, main=a)");
            caller.endPlot();
            caller.runOnly();
            caller.showPlot(file);
        } catch (IOException ex) {
            Logger.getLogger(FFNN.class.getName()).log(Level.SEVERE, null, ex);
        }
        
        
        Vector forecastError = tsnn.generalizationNN(validationSet,numOfInputUnit, numOfHiddenUnit, numOfOutputUnit,eta, alpha);
        double [] SE = (double[]) forecastError.elementAt(0);
        double [] absE = (double[]) forecastError.elementAt(1);
        double [] absPE = (double[]) forecastError.elementAt(2);
        double RMSEValidation = tsnn.calculateForecastError(SE);
        double MAE = tsnn.calculateMAE(absE);
        double MAPE = tsnn.calculateMAPE(absPE);      
        //System.out.println("Forecast RMSE = "+RMSEValidation);
        double [] result = new double[3];
        result[0] = RMSEValidation;
        result[1] = MAE;
        result[2] = MAPE;
        return result;
    }

    public double[] FFNN(double[][] V, double[][] W) {
        TimeSeriesNN tsnn = new TimeSeriesNN(V, W);
        Vector dataSet = getDataSetNYB();
        tsnn.setMinMax((double[]) dataSet.elementAt(3));
        
        double[] trainingSet = tsnn.normalizeData((double[]) dataSet.elementAt(0));
        double[] testingSet = tsnn.normalizeData((double[]) dataSet.elementAt(1));
        double[] validationSet = tsnn.normalizeData((double[]) dataSet.elementAt(2));

        double[] RMSE = tsnn.TrainingNN(weightInitialization, trainingSet, testingSet, numOfInputUnit, numOfHiddenUnit, numOfOutputUnit, eta, alpha, maxEpoch, maxError);
       
         try {
            RCaller caller = new RCaller();
            caller.setRscriptExecutable("/usr/bin/Rscript");
            caller.cleanRCode();
            File file;
            String[] arr = new String[1];
            arr[0] = "New York Birth";
            file = caller.startPlot();
            caller.addDoubleArray("RMSE.ffnn", RMSE);
            caller.addStringArray("arr", arr);
            caller.addRCode("a = arr[1]");
            caller.addRCode("plot.ts(RMSE.ffnn, main=a)");
            caller.endPlot();
            caller.runOnly();
            caller.showPlot(file);
        } catch (IOException ex) {
            Logger.getLogger(FFNN.class.getName()).log(Level.SEVERE, null, ex);
        }
         
        
        Vector forecastError = tsnn.generalizationNN(validationSet,numOfInputUnit, numOfHiddenUnit, numOfOutputUnit,eta, alpha);
        double [] SE = (double[]) forecastError.elementAt(0);
        double [] absE = (double[]) forecastError.elementAt(1);
        double [] absPE = (double[]) forecastError.elementAt(2);
        double RMSEValidation = tsnn.calculateForecastError(SE);
        double MAE = tsnn.calculateMAE(absE);
        double MAPE = tsnn.calculateMAPE(absPE);      
        //System.out.println("Forecast RMSE = "+RMSEValidation);
        double [] result = new double[3];
        result[0] = RMSEValidation;
        result[1] = MAE;
        result[2] = MAPE;
        return result;
    }

    public double [] HybridRNN(double[][] V, double[][] W, double[][] U) {
        TimeSeriesHybridRNN tsnn = new TimeSeriesHybridRNN(V, W, U);
        Vector dataSet = getDataSetNYB();
        tsnn.setMinMax((double[]) dataSet.elementAt(3));
        double[] trainingSet = tsnn.normalizeData((double[]) dataSet.elementAt(0));
        double[] testingSet = tsnn.normalizeData((double[]) dataSet.elementAt(1));
        double[] validationSet = tsnn.normalizeData((double[]) dataSet.elementAt(2));

        double[] RMSE = tsnn.TrainingNN(weightInitialization, trainingSet, testingSet, numOfInputUnit, numOfHiddenUnit, numOfOutputUnit, eta, alpha, maxEpoch, maxError);
        
        try {
         RCaller caller = new RCaller();
         caller.setRscriptExecutable("/usr/bin/Rscript");
         caller.cleanRCode();
         File file;
         String [] arr = new String[1];
         arr[0] = "New York Birth";
         file = caller.startPlot();
         caller.addDoubleArray("RMSE.hybrid.rnn", RMSE);
         caller.addStringArray("arr", arr);
         caller.addRCode("a = arr[1]");
         caller.addRCode("plot.ts(RMSE.hybrid.rnn, main=a)");
         caller.endPlot();
         caller.runOnly();
         caller.showPlot(file);
         } catch (IOException ex) {
         Logger.getLogger(FFNN.class.getName()).log(Level.SEVERE, null, ex);
         } 
        

        Vector forecastError = tsnn.generalizationNN(validationSet,numOfInputUnit, numOfHiddenUnit, numOfOutputUnit,eta, alpha);
        double [] SE = (double[]) forecastError.elementAt(0);
        double [] absE = (double[]) forecastError.elementAt(1);
        double [] absPE = (double[]) forecastError.elementAt(2);
        double RMSEValidation = tsnn.calculateForecastError(SE);
        double MAE = tsnn.calculateMAE(absE);
        double MAPE = tsnn.calculateMAPE(absPE);      
        double [] result = new double[3];
        result[0] = RMSEValidation;
        result[1] = MAE;
        result[2] = MAPE;
        return result;
    }
   
    public static void main(String[] args) {
        int I = 12;
        int J = 18;
        int K = 12;
        double eta = 0.25;
        double alpha = 0.25;
        int maxEpoch = 500;
        double maxError = 0.5;
        int choice = 1;
        double[][] experiment = new double[30][12];

        for (int i = 0; i < 1; i++) {
            WeightsInitialization weight0 = new WeightsInitialization();
            Comparison result = new Comparison(choice, I, J, K, eta, alpha, maxEpoch, maxError);
            double[][] V = weight0.useRandomWeight(I, J);
            double[][] W = weight0.useRandomWeight(J, K);
            double[][] U = weight0.useRandomWeightForU(J, J);
            double [] E_RNN = result.RNN(V, W, U);
            double [] E_RNNH = result.HybridRNN(V, W, U);
            double [] E_FFNN = result.FFNN(V, W);
            double [] E_FFNNH = result.HybridFFNN(V, W);
            experiment[i][0] = E_FFNN [0];
            experiment[i][1] = E_FFNNH [0];
            experiment[i][2] = E_RNN[0];
            experiment[i][3] = E_RNNH[0];
            experiment[i][4] = E_FFNN [1];
            experiment[i][5] = E_FFNNH [1];
            experiment[i][6] = E_RNN[1];
            experiment[i][7] = E_RNNH[1];
            experiment[i][8] = E_FFNN [2];
            experiment[i][9] = E_FFNNH [2];
            experiment[i][10] = E_RNN[2];
            experiment[i][11] = E_RNNH[2];            
        }

        
        try (
                PrintStream output = new PrintStream(new File("/Users/Ega/Desktop/NYB1/E1.txt"));) {

            for (int i = 0; i < 30; i++) {
                String sc = "";
                for (int j = 0; j < experiment[i].length; j++) {
                    if (j < 11) {
                        sc += Double.toString(experiment[i][j]) + ", ";
                    } else {
                        sc += Double.toString(experiment[i][j]);
                    }

                }
                output.println(sc);
            }
            output.close();

        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        
    }
}
