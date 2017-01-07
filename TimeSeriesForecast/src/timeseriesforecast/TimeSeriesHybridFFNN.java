/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package timeseriesforecast;

import java.util.Vector;
import rcaller.RCaller;
import rcaller.RCode;

/**
 *
 * @author Ega
 */
public class TimeSeriesHybridFFNN {

    private double[][] weightsV;
    private double[][] weightsW;
    private double minValue;
    private double maxValue;
    
    public TimeSeriesHybridFFNN(double [][] V, double [][]W){
        this.weightsV = V;
        this.weightsW = W;
    }

    public double[] TrainingNN(int choiceW, double[] trainingSet, double[] testingSet,int numOfInputUnit, int numOfHiddenUnit, int numOfOutputUnit, double eta, double alpha, int maxEpoch, double maxError) {
        /*
         * Learning in one-step forecast or multi-step forecast
         */
        FFNN ffnn = new FFNN(numOfInputUnit, numOfHiddenUnit, numOfOutputUnit, eta, alpha, getWeightsV(), getWeightsW());
        ffnn.initializeRandomWeight(choiceW);
        System.out.println("");
        ffnn.showWeights(ffnn.getV());
        System.out.println("");
        ffnn.showWeights(ffnn.getW());
        System.out.println("");
        boolean stopCondition = false;
        double forecastError = 9999999.99999;
        int epoch = 0;
        
        double[] SE = new double[trainingSet.length - numOfOutputUnit - numOfInputUnit];
        
        double[] RMSEtemp = new double[maxEpoch + 1];
        double[] output = new double[numOfOutputUnit];
        
        while (stopCondition != true) {

            for (int l = 0; l < (trainingSet.length - numOfOutputUnit - numOfInputUnit); l++) {
                double[] dataPoint = new double[numOfInputUnit];
                double[] target = new double[numOfOutputUnit];
                //set value for input units
                for (int p = 0; p < numOfInputUnit; p++) {
                    dataPoint[p] = trainingSet[p + l];
                }
                //set value for target unit  for one step forecast or multi step forecast
                for (int p = 0; p < numOfOutputUnit; p++) {
                    if ((numOfInputUnit + p + l) < trainingSet.length) {
                        target[p] = trainingSet[numOfInputUnit + p + l];
                    } else {
                        break;
                    }
                }

                ffnn.gradientDescent(dataPoint, target, eta, alpha);
                setWeightsV(ffnn.getV());
                setWeightsW(ffnn.getW());
            }

            //SE = TestingNN(testingSet, numOfInputUnit, numOfHiddenUnit, numOfOutputUnit, eta, alpha);
            
            RMSEtemp[epoch] = calculateForecastError(SE);
            epoch += 1;
            if(epoch>maxEpoch){
                stopCondition= true;
            }
            
        }
        double[] RMSE = new double[epoch];
        for(int i=0; i<epoch; i++){
            RMSE[i] = RMSEtemp[i];
        }
        System.out.println("rmse testing" +RMSE[epoch-1]);
        return RMSE;
    }

    public double [] TestingNN(double[] testingSet, int numOfInputUnit, int numOfHiddenUnit, int numOfOutputUnit, double eta, double alpha) {
        /*
         * Testing one-step forecast or multi-step forecast
         */
        
        FFNN ffnn = new FFNN(numOfInputUnit, numOfHiddenUnit, numOfOutputUnit, eta, alpha, getWeightsV(), getWeightsW());
        ffnn.setV(getWeightsV());
        ffnn.setW(getWeightsW());
        int L = (testingSet.length - numOfOutputUnit - numOfInputUnit);
        double[][] outputSetD = new double[L][numOfOutputUnit];
        double[][] targetSetD = new double[L][numOfOutputUnit];
        double[][] targetSet = new double[L][numOfOutputUnit];
        double[] output = new double[numOfOutputUnit];
        double[] SE = new double[L];
        double [][] errorForInput = new double[L][numOfOutputUnit];
        for (int l = 0; l < L; l++) {
            double[] dataPoint = new double[numOfInputUnit];
            double[] target = new double[numOfOutputUnit];
            //set value for input units
            for (int p = 0; p < numOfInputUnit; p++) {
                dataPoint[p] = testingSet[p + l];
            }
            output = ffnn.feedforward(dataPoint);
            
            //set testing set
            for (int p = 0; p < numOfOutputUnit; p++) {
                if ((numOfInputUnit + p + l) < testingSet.length) {
                    targetSet[l][p] = testingSet[numOfInputUnit + p + l];
                    target[p] = targetSet[l][p];
                } else {
                    break;
                }
            }
            
            output = ffnn.feedforward(dataPoint);
            double[] outputD = denormalizeData(output);
            double[] targetD = denormalizeData(target);
            for (int p = 0; p < numOfOutputUnit; p++) {
                outputSetD[l][p] = outputD[p];
                targetSetD[l][p] = targetD[p];
                
            }
            
            for (int p = 0; p < numOfOutputUnit; p++) {
                outputSetD[l][p] = outputD[p];
                targetSetD[l][p] = targetD[p];
                errorForInput[l][p] =  outputSetD[l][p] - targetSetD[l][p];
                //System.out.println("output-t-"+outputSetD[l][p]+" target-t-"+targetSetD[l][p]);
                //System.out.println("error for input"+errorForInput[l][p]);
            }
            
        }
        
         
        double [][] linearOutput = new double[outputSetD.length][outputSetD[0].length];
        
        
        
         for(int i = 0; i<errorForInput[0].length; i++){
            ARIMA arima = new ARIMA();
            double [] arimaOutput = new double [linearOutput.length];
            double [] arimaInput = new double [linearOutput.length];
            
            
            for(int j = 0; j< linearOutput.length; j++){
               arimaInput[j] = errorForInput[j][i];
            }
            arimaOutput = arima.getPredictionValueOnInputError(arimaInput);
            
            //System.out.println("Arima output length" + arimaOutput.length);
            
            for(int j = 0; j< linearOutput.length; j++){
                linearOutput[j][i] = arimaOutput[j];                
               //System.out.println("arima Output "+arimaOutput[j]);
            }
        }
        
        // combine output of NN and ARIMA
        double[][] hybridOutput = new double[testingSet.length][numOfOutputUnit];
        for(int j = 0; j< outputSetD.length; j++){
            for(int i =0; i<outputSetD[0].length; i++){
                if(errorForInput[j][i] < 0 && linearOutput[j][i] <0){
                    hybridOutput[j][i] = outputSetD[j][i] - linearOutput[j][i];
                }
                else
                    hybridOutput[j][i] = outputSetD[j][i] + linearOutput[j][i];
            }
            SE[j] = ffnn.calculateSE(hybridOutput[j], targetSetD[j]);
        }
         
                
        return SE;
    }
    
    public double[] generalizationNN(double[] validationSet, int numOfInputUnit, int numOfHiddenUnit, int numOfOutputUnit, double eta, double alpha) {
        /*
         * one-step forecast or multi-step forecast
         */
        FFNN ffnn = new FFNN(numOfInputUnit, numOfHiddenUnit, numOfOutputUnit, eta, alpha, getWeightsV(), getWeightsW());
        ffnn.setV(getWeightsV());
        ffnn.setW(getWeightsW());

        int L = validationSet.length ;
        System.out.println("L"+L);
        double[][] outputSetD = new double[L][numOfOutputUnit];
        double[][] targetSetD = new double[L][numOfOutputUnit];
        double[][] targetSet = new double[L][numOfOutputUnit];
        double[] output = new double[numOfOutputUnit];
        double[] SE = new double[L];
        double [][] errorForInput = new double[L][numOfOutputUnit];
        for (int l = 0; l < (validationSet.length - numOfOutputUnit - numOfInputUnit); l++) {
            double[] dataPoint = new double[numOfInputUnit];
            double[] target = new double[numOfOutputUnit];
            //set value for input units
            for (int p = 0; p < numOfInputUnit; p++) {
                dataPoint[p] = validationSet[p + l];
            }
            output = ffnn.feedforward(dataPoint);
            
            //set testing set
            for (int p = 0; p < numOfOutputUnit; p++) {
                if ((numOfInputUnit + p + l) < validationSet.length) {
                    targetSet[l][p] = validationSet[numOfInputUnit + p + l];
                    target[p] = targetSet[l][p];
                } else {
                    break;
                }
            }
            
            output = ffnn.feedforward(dataPoint);
            double[] outputD = denormalizeData(output);
            double[] targetD = denormalizeData(target);
            for (int p = 0; p < numOfOutputUnit; p++) {
                outputSetD[l][p] = outputD[p];
                targetSetD[l][p] = targetD[p];
                errorForInput[l][p] = targetSetD[l][p]- outputSetD[l][p];
                //System.out.println("output-t-"+outputSetD[l][p]+" target-t-"+targetSetD[l][p]);
                //System.out.println("error for input"+errorForInput[l][p]);
                
            }
            
        }
        
        
        double [][] linearOutput = new double[outputSetD.length][outputSetD[0].length];
        
        //calculate forecast using ARIMA
        //System.out.println("errorForInput[0].length "+errorForInput[0].length);
        //System.out.println("linearOutput.length "+linearOutput.length);
        for(int i = 0; i<errorForInput[0].length; i++){
            ARIMA arima = new ARIMA();
            double [] arimaOutput = new double [linearOutput.length];
            double [] arimaInput = new double [linearOutput.length];
            
            
            for(int j = 0; j< linearOutput.length; j++){
               arimaInput[j] = errorForInput[j][i];
            }
            arimaOutput = arima.getPredictionValueOnInputError(arimaInput);
            
            //System.out.println("Arima output length" + arimaOutput.length);
            
            for(int j = 0; j< linearOutput.length; j++){
                linearOutput[j][i] = arimaOutput[j];
                
               //System.out.println("arima Output "+arimaOutput[j]);
            }
        }
        
        // combine output of NN and ARIMA
        double[][] hybridOutput = new double[validationSet.length][numOfOutputUnit];
        for(int j = 0; j< outputSetD.length; j++){
            for(int i =0; i<outputSetD[0].length; i++){
                if(errorForInput[j][i] < 0){
                    hybridOutput[j][i] = outputSetD[j][i] - linearOutput[j][i];
                }
                else
                    hybridOutput[j][i] = outputSetD[j][i] + linearOutput[j][i];
            }
            SE[j] = ffnn.calculateSE(hybridOutput[j], targetSetD[j]);
        }        
        return SE;
    }

    public double calculateSEHybrid(double[] output, double[] target) {
        double SE = 0;
        int K = output.length;
        for (int k = 0; k < K; k++) {
            double e = target[k] - output[k];
            SE += e * e;
            
        }
        SE = SE/K;        
        return SE;
    }
    
    public double calculateForecastError(double[] squaredError) {
        double SSE = 0.0;
        for (int i = 0; i < squaredError.length; i++) {
            SSE += squaredError[i];
        }
        double MSE = SSE / squaredError.length;
        double RMSE = Math.sqrt(MSE);
        //System.out.println("rmse " + RMSE);
        return RMSE;
    }
    
    public void setMinMax(double[] dataSet) {
        RCaller caller = new RCaller();
        caller.setRscriptExecutable("/usr/local/bin/Rscript");

        RCode code = new RCode();
        code.clear();

        /*get maximum and minimum value from data set*/
        code.addDoubleArray("dataSet", dataSet);
        code.addRCode("maxVal <- max(dataSet)");
        code.addRCode("minVal <- min(dataSet)");
        code.addRCode("results <-list(max = maxVal, min=minVal)");
        caller.setRCode(code);
        caller.runAndReturnResult("results");
        double[] max = caller.getParser().getAsDoubleArray("max");
        double[] min = caller.getParser().getAsDoubleArray("min");
        setMaxValue(max[0]);
        setMinValue(min[0]);
    }

    public double[] normalizeData(double[] dataSet) {
        /*
         * Normalize Data Set between 0.2 and 0.8
         */
        double[] normalizedData = new double[dataSet.length];
        double maxD = getMaxValue();
        double minD = getMinValue();
        /*Normalized Data*/
        for (int i = 0; i < dataSet.length; i++) {
            normalizedData[i] = (0.8 - 0.2) * ((dataSet[i] - minD) / (maxD - minD)) + 0.2;
        }
        return normalizedData;
    }

    public double[] denormalizeData(double[] normalizedData) {
        /*
         * get original value of data set
         */
        double[] originalData = new double[normalizedData.length];
        double maxD = getMaxValue();
        double minD = getMinValue();
        /*Denormalized Data*/
        for (int i = 0; i < normalizedData.length; i++) {

            originalData[i] = (((normalizedData[i] - 0.2) / (0.8 - 0.2)) * (maxD - minD)) + minD;
        }
        return originalData;
    }

    public double[] normalizeDataUsingSigmoid(double[] dataSet) {
        /*
         * Normalize Data Set between 0.2 and 0.8
         */
        double[] normalizedData = new double[dataSet.length];

        /*Normalized Data*/
        for (int i = 0; i < dataSet.length; i++) {
            normalizedData[i] = 1 / (1 + Math.exp(-1 * dataSet[i]));
        }
        return normalizedData;
    }

    public double[] denormalizeDataInverseSigmoid(double[] normalizedData) {
        /*
         * get original value of data set
         */
        double[] originalData = new double[normalizedData.length];
        double maxD = getMaxValue();
        double minD = getMinValue();
        /*Denormalized Data*/
        for (int i = 0; i < normalizedData.length; i++) {
            //originalData[i] = (inverseSigmoid((normalizedData[i] - 0.2)/(0.8-0.2))*(maxD-minD)) + minD;
            originalData[i] = inverseSigmoid(normalizedData[i]);
        }
        return originalData;
    }

    /**
     * @return the weightsV
     */
    public double inverseSigmoid(double x) {
        double inverse = Math.log(x) - Math.log(1 - x);
        return inverse;
    }

    public double[][] getWeightsV() {
        return weightsV;
    }

    /**
     * @param weightsV the weightsV to set
     */
    public void setWeightsV(double[][] weightsV) {
        this.weightsV = weightsV;
    }

    /**
     * @return the weightsW
     */
    public double[][] getWeightsW() {
        return weightsW;
    }

    /**
     * @param weightsW the weightsW to set
     */
    public void setWeightsW(double[][] weightsW) {
        this.weightsW = weightsW;
    }

    /**
     * @return the minValue
     */
    public double getMinValue() {
        return minValue;
    }

    /**
     * @param minValue the minValue to set
     */
    public void setMinValue(double minValue) {
        this.minValue = minValue;
    }

    /**
     * @return the maxValue
     */
    public double getMaxValue() {
        return maxValue;
    }

    /**
     * @param maxValue the maxValue to set
     */
    public void setMaxValue(double maxValue) {
        this.maxValue = maxValue;
    }
}
