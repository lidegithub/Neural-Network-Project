/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package timeseriesforecast;

import java.io.File;
import java.io.IOException;
import java.util.Vector;
import java.util.logging.Level;
import java.util.logging.Logger;
import rcaller.RCaller;
import rcaller.RCode;

/**
 *
 * @author Ega
 */
public class ComparisonSouvenirDS {

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
    
    public ComparisonSouvenirDS(int weightInitialization, int numOfInputUnit, int numOfHiddenUnit, int numOfOutputUnit, double eta, double alpha, int maxEpoch, double maxError) {
        this.numOfInputUnit = numOfInputUnit;
        this.numOfHiddenUnit = numOfHiddenUnit;
        this.numOfOutputUnit = numOfOutputUnit;
        this.eta = eta;
        this.alpha = alpha;
        this.maxEpoch = maxEpoch;
        this.maxError = maxError;
        this.weightInitialization = weightInitialization;
    }

    public Vector getDataSetSouvenir() {
        Vector results = new Vector();
        double[] trainingSet;
        double[] testingSet;
        double[] validationSet;
        try {
            /* Creating a RCaller */
            RCaller caller = new RCaller();
            caller.setRscriptExecutable("/usr/local/bin/Rscript");

            /* Creating a source code */
            RCode code = new RCode();
            code.clear();

            // add libraries needed to load data set
            code.addRCode("library(zoo)");
            code.addRCode("library(timeSeries)");
            code.addRCode("library(rdatamarket)");

            //get data for training, testing and validation set
            code.addRCode("dminit(\"be0123ef782e49348a7ed53c2444c08c\")");
            code.addRCode("dataSet <- dmlist(\"22mh\")");
            code.addRCode("trainingSet <- dataSet[1:60,2]");
            code.addRCode("testingSet <- dataSet[61:84,2]");
            code.addRCode("validationSet <- dataSet[61:84,2]");
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

    public double HybridFFNN(double [][]V, double [][]W) {
        TimeSeriesHybridFFNN tsnn = new TimeSeriesHybridFFNN(V,W);
        Vector dataSet = getDataSetSouvenir();
        tsnn.setMinMax((double[]) dataSet.elementAt(3));

        double[] trainingSet = tsnn.normalizeData((double[]) dataSet.elementAt(0));
        double[] testingSet = tsnn.normalizeData((double[]) dataSet.elementAt(1));
        double[] validationSet = tsnn.normalizeData((double[]) dataSet.elementAt(2));

        double[] RMSE = tsnn.TrainingNN(weightInitialization, trainingSet, testingSet, numOfInputUnit, numOfHiddenUnit, numOfOutputUnit, eta, alpha, maxEpoch, maxError);
        /*
         *         try {
            RCaller caller = new RCaller();
            caller.setRscriptExecutable("/usr/local/bin/Rscript");
            caller.cleanRCode();
            File file;
            String[] arr = new String[1];
            arr[0] = "Souvenir Sales";
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
         */
       
        double[] SE = tsnn.generalizationNN(validationSet, numOfInputUnit, numOfHiddenUnit, numOfOutputUnit, eta, alpha);
        double RMSEValidation = tsnn.calculateForecastError(SE);
        //System.out.println("Forecast RMSE = " + RMSEValidation);
        return RMSEValidation;
    }
    
    public double RNN(double [][]V, double [][] W, double [][] U) {
        TimeSeriesRNN tsnn = new TimeSeriesRNN(V, W, U);
        Vector dataSet = getDataSetSouvenir();
        tsnn.setMinMax((double[]) dataSet.elementAt(3));
        System.out.println("max "+tsnn.getMaxValue());
        System.out.println("min "+tsnn.getMinValue());
        double[] trainingSet = tsnn.normalizeData((double[]) dataSet.elementAt(0));
        double[] testingSet = tsnn.normalizeData((double[]) dataSet.elementAt(1));        
        double[] validationSet = tsnn.normalizeData((double[]) dataSet.elementAt(2));        
        double[] RMSE = tsnn.TrainingNN(weightInitialization, trainingSet, testingSet, numOfInputUnit, numOfHiddenUnit, numOfOutputUnit,eta, alpha, maxEpoch, maxError);
        try {
            RCaller caller = new RCaller();
            caller.setRscriptExecutable("/usr/local/bin/Rscript");
            caller.cleanRCode();
            File file;
            String [] arr = new String[1];
            arr[0] = "Souvenir Sales";
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
        double [] SE = tsnn.generalizationNN(validationSet,numOfInputUnit, numOfHiddenUnit, numOfOutputUnit,eta, alpha);
        double RMSEValidation = tsnn.calculateForecastError(SE);
        //System.out.println("Forecast RMSE = "+RMSEValidation);
        return RMSEValidation;
    }
    
    public double FFNN( double [][] V, double [][] W) {
        TimeSeriesNN tsnn = new TimeSeriesNN(V, W);
        Vector dataSet = getDataSetSouvenir();
        tsnn.setMinMax((double[]) dataSet.elementAt(3));
        System.out.println("max "+tsnn.getMaxValue());
        System.out.println("min "+tsnn.getMinValue());
        double[] trainingSet = tsnn.normalizeData((double[]) dataSet.elementAt(0));
        double[] testingSet = tsnn.normalizeData((double[]) dataSet.elementAt(1));        
        double[] validationSet = tsnn.normalizeData((double[]) dataSet.elementAt(2));
        
        double[] RMSE = tsnn.TrainingNN(weightInitialization, trainingSet, testingSet, numOfInputUnit, numOfHiddenUnit, numOfOutputUnit,eta, alpha, maxEpoch, maxError);
        try {
            RCaller caller = new RCaller();
            caller.setRscriptExecutable("/usr/local/bin/Rscript");
            caller.cleanRCode();
            File file;
            String [] arr = new String[1];
            arr[0] = "Souvenir Sales";
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
        double [] SE = tsnn.generalizationNN(validationSet,numOfInputUnit, numOfHiddenUnit, numOfOutputUnit,eta, alpha);
        double RMSEValidation = tsnn.calculateForecastError(SE);
        //System.out.println("Forecast RMSE = "+RMSEValidation);
        return RMSEValidation;

    }
    
    public double HybridRNN(double [][] V, double [][] W, double [][] U) {
        TimeSeriesHybridRNN tsnn = new TimeSeriesHybridRNN(V, W, U);
        Vector dataSet = getDataSetSouvenir();
        tsnn.setMinMax((double[]) dataSet.elementAt(3));
        System.out.println("max "+tsnn.getMaxValue());
        System.out.println("min "+tsnn.getMinValue());
        double[] trainingSet = tsnn.normalizeData((double[]) dataSet.elementAt(0));
        double[] testingSet = tsnn.normalizeData((double[]) dataSet.elementAt(1));        
        double[] validationSet = tsnn.normalizeData((double[]) dataSet.elementAt(2));
           
        double[] RMSE = tsnn.TrainingNN(weightInitialization, trainingSet, testingSet, numOfInputUnit, numOfHiddenUnit, numOfOutputUnit,eta, alpha, maxEpoch, maxError);
        /*
         *try {
            RCaller caller = new RCaller();
            caller.setRscriptExecutable("/usr/local/bin/Rscript");
            caller.cleanRCode();
            File file;
            String [] arr = new String[1];
            arr[0] = "Souvenir Sales";
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
         */
        
        double [] SE = tsnn.generalizationNN(validationSet,numOfInputUnit, numOfHiddenUnit, numOfOutputUnit,eta, alpha);
        double RMSEValidation = tsnn.calculateForecastError(SE);
        //System.out.println("Forecast RMSE = "+RMSEValidation);
        return RMSEValidation;
    }
    
    public static void main(String[] args) {
        int I = 12;
        int J = 12;
        int K = 12;
        double eta = 0.25;
        double alpha = 0.25;
        int maxEpoch = 10000;
        double maxError = 0.5;
        int choice = 1; 
        WeightsInitialization weight0 = new WeightsInitialization();
        ComparisonSouvenirDS result = new ComparisonSouvenirDS (choice,I,J,K,eta, alpha, maxEpoch, maxError);
        double [][] V = weight0.useRandomWeight(I, J);
        double [][] W = weight0.useRandomWeight(J, K);
        double [][] U = weight0.useRandomWeightForU(J, J);
        double RMSE_RNN = result.RNN(V, W, U);        
        //double RMSE_RNNH = result.HybridRNN(V, W, U);
        double RMSE_FFNN = result.FFNN(V,W);       
        //double RMSE_FFNNH = result.HybridFFNN(V,W);        
        
        System.out.println("RMSE FFNN = " +RMSE_FFNN);
        //System.out.println("RMSE FFNN Hybrid  = " +RMSE_FFNNH);
        System.out.println("RMSE RNN = " +RMSE_RNN);
       // System.out.println("RMSE RNN Hybrid = " +RMSE_RNNH);
        
    }
}
