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
public class ForecastHybridRNN {

    private int numOfInputUnit;
    private int numOfHiddenUnit;
    private int numOfOutputUnit;
    private double[][] V;
    private double[][] W;
    private double eta;
    private double alpha;
    private int maxEpoch;
    private double maxError;
    private int weightInitialization;

    public ForecastHybridRNN(){
    
    }
    
    public ForecastHybridRNN(int weightInitialization, int numOfInputUnit, int numOfHiddenUnit, int numOfOutputUnit, double eta, double alpha, int maxEpoch, double maxError) {
        this.numOfInputUnit = numOfInputUnit;
        this.numOfHiddenUnit = numOfHiddenUnit;
        this.numOfOutputUnit = numOfOutputUnit;
        this.eta = eta;
        this.alpha = alpha;
        this.V = new double[numOfHiddenUnit][numOfInputUnit + 1];
        this.W = new double[numOfOutputUnit][numOfHiddenUnit + 1];
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
            double [] allData = caller.getParser().getAsDoubleArray("data");
            results.addElement(trainingSet);
            results.addElement(testingSet);
            results.addElement(validationSet);
            results.addElement(allData);
        } catch (Exception ex) {
            System.out.println(ex.getMessage());
        }
        return results;
    }

    public double trainNN(double [][] V, double [][] W, double [][] U) {
        TimeSeriesHybridRNN tsnn = new TimeSeriesHybridRNN(V, W, U);
        Vector dataSet = getDataSetNYB();
        tsnn.setMinMax((double[]) dataSet.elementAt(3));
        System.out.println("max "+tsnn.getMaxValue());
        System.out.println("min "+tsnn.getMinValue());
        double[] trainingSet = tsnn.normalizeData((double[]) dataSet.elementAt(0));
        double[] testingSet = tsnn.normalizeData((double[]) dataSet.elementAt(1));        
        double[] validationSet = tsnn.normalizeData((double[]) dataSet.elementAt(2));
        
        setV(tsnn.getWeightsV());
        setW(tsnn.getWeightsW());
        
        double[] RMSE = tsnn.TrainingNN(weightInitialization, trainingSet, testingSet, numOfInputUnit, numOfHiddenUnit, numOfOutputUnit,eta, alpha, maxEpoch, maxError);
        try {
            RCaller caller = new RCaller();
            caller.setRscriptExecutable("/usr/local/bin/Rscript");
            caller.cleanRCode();
            File file;
            String [] arr = new String[1];
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
        double [] SE = tsnn.generalizationNN(validationSet,numOfInputUnit, numOfHiddenUnit, numOfOutputUnit,eta, alpha);
        double RMSEValidation = tsnn.calculateForecastError(SE);
        System.out.println("Forecast RMSE = "+RMSEValidation);
        return RMSEValidation;
    }
    
   
    public static void main(String[] args) {
        int I = 12;
        int J = 6;
        int K = 1;
        double eta = 0.25;
        double alpha = 0.25;
        int maxEpoch = 500;
        double maxError = 0.5;
        int choice = 1;
        ForecastHybridRNN rnn = new ForecastHybridRNN(choice,I,J,K,eta, alpha, maxEpoch, maxError);
        WeightsInitialization weight0 = new WeightsInitialization();
        double [][] V = weight0.useRandomWeight(I, J);
        double [][] W = weight0.useRandomWeight(J, K);
        double [][] U = weight0.useRandomWeightForU(J, J);
        double RMSE = rnn.trainNN(V, W, U);       
        System.out.println("rmse hybrid rnn = "+RMSE);
    }

    /**
     * @return the V
     */
    public double[][] getV() {
        return V;
    }

    /**
     * @param V the V to set
     */
    public void setV(double[][] V) {
        this.V = V;
    }

    /**
     * @return the W
     */
    public double[][] getW() {
        return W;
    }

    /**
     * @param W the W to set
     */
    public void setW(double[][] W) {
        this.W = W;
    }
}
