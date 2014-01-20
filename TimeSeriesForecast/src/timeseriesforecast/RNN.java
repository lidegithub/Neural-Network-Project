/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package timeseriesforecast;

/**
 *
 * @author Ega
 */
public class RNN {
    private double[][] V;       // Weights between input and hidden layer
    private double[][] W;        // Weights between hidden and ouput layer
    private double [][] U;
    private double[][] deltaV;
    private double[][] deltaW;
    private double[][] deltaU;
    private int numOfInputUnit;
    private int numOfHiddenUnit;
    private int numOfOutputUnit;
    //private int epoch;
    private double eta;
    private double alpha;
    private double [] hiddenUnitOutput;

    public RNN(int I, int J, int K, double eta, double alpha, double [][] V, double [][] W, double [][] U) {
        this.numOfInputUnit = I;
        this.numOfHiddenUnit = J;
        this.numOfOutputUnit = K;
        this.V = V;
        this.deltaV = new double[J][I + 1];
        this.W = W;
        this.deltaW = new double[K][J + 1];
        this.U = U;
        //System.out.println("Initial U"+U.length+"x"+U[0].length);
        this.deltaU = new double[J][J];
        this.eta = eta;
        this.alpha = alpha;
        this.hiddenUnitOutput = new double [J];
    }
    
    public void initializeInputFromHiddenNeuronRNN(int J){
        double[] hiddenOutput = new double [J];
        for (int j = 0; j<J; j++){
            hiddenOutput[j] = 0;
        }
        setHiddenUnitOutput(hiddenOutput);
    }

    public void initializeRandomWeight(int choice) {
        /*
         * if choice = 1 generate random number between -0.5 and 0.5 for weights initialization        
         */     
        
        int I = numOfInputUnit;
        int J = numOfHiddenUnit;
        int K = numOfOutputUnit;

        //initialization for delta
        for (int j = 0; j < J; j++) {
                for (int i = 0; i <= I; i++) {
                    deltaV[j][i] = 0.0;
                }
            }
        
        for (int j = 0; j < J; j++) {
                for (int i = 0; i < J; i++) {
                    deltaU[j][i] = 0.0;
                }
            }
        
        for (int k = 0; k < K; k++) {
            for (int j = 0; j <= J; j++) {                
                deltaW[k][j] = 0.0;
            }
        }
    }

    public void showWeights(double[][] Weights) {
        for (int j = 0; j < Weights.length; j++) {
            for (int i = 0; i < Weights[0].length; i++) {
                int x = j + 1;
                int y = i + 1;
                System.out.printf(x + "," + y + ": " + Weights[j][i] + " ");
            }
            System.out.println("");
        }
    }

    public void showVector(double[] vector) {
        for (int i = 0; i < vector.length; i++) {
            System.out.print(" " + vector[i]);
        }
    }

    private void showTrainingSet(double[][] dataSet) {
        int m = dataSet.length;
        int n = dataSet[0].length;
        for (int j = 0; j < m; j++) {
            for (int i = 0; i < n; i++) {
                System.out.print(dataSet[j][i] + " ");
            }
            System.out.println("");
        }
    }

    public double[] forward(double[] dataPoint) {
        double[] output = new double [numOfOutputUnit];
        
        //forward propagation from input to hidden layer
        double[] input = createAugmentedVector(dataPoint); //z
        double[][] v = getV();
        double[][] u = getU();
        
        double [] hiddenOutput = getHiddenUnitOutput();
        //System.out.println("H" +hiddenOutput.length);
        //System.out.println("I" +input.length);
        double[] sup = sumOfProductHidden(input,hiddenOutput,v, u);
        double[] nonLinear = calculateNonLinearVector(sup);

        //forward propagation from hidden to output layer
        double[] inputForOutput = createAugmentedVector(nonLinear); //y
        double[][] w = getW();

        double[] supOutput = sumOfProduct(inputForOutput, w);
        output = calculateNonLinearVector(supOutput); //O
        return output;

    }
    
    public double[] gradientDescent(double[] dataPoint, double[] target, double eta, double alpha) {

        //forward propagation from input to hidden layer
        double[] input = createAugmentedVector(dataPoint); //z
        double[][] v = getV();
        double[][] u = getU();
        double [] hiddenOutput = getHiddenUnitOutput();
        double[] sup = sumOfProductHidden(input,hiddenOutput, v, u);
        double[] nonLinear = calculateNonLinearVector(sup);
        setHiddenUnitOutput(nonLinear);

        //forward propagation from hidden to output layer
        double[] inputForOutput = createAugmentedVector(nonLinear); //y
        double[][] w = getW();

        double[] supOutput = sumOfProduct(inputForOutput, w);
        double[] output = calculateNonLinearVector(supOutput); //O

        //backward propagation

        double[] gammaO = calculateGammaO(target, output);
        double[][] deltaWeight0 = getDeltaWeightO(gammaO, inputForOutput, eta);
        updateWeight0(alpha, deltaWeight0);

        double[] gammaY = calculateGammaY(gammaO, inputForOutput, w);
        double[][] deltaWeightY = getDeltaWeightY(gammaY, input, eta);
        //System.out.println("W " +w.length+"x"+w[0].length);
        //System.out.println("V " +v.length+"x"+v[0].length);
        //System.out.println("U " +u.length+"x"+u[0].length);
        //System.out.println("Gamma O"+gammaO.length);
        //System.out.println("Gamma Y"+gammaY.length);
        //System.out.println("deltaWeight Y " + deltaWeightY.length+ "x" +deltaWeightY[0].length);
        updateWeightY(alpha, deltaWeightY);
        
        double[][] deltaWeightU = getDeltaWeightY(gammaY, getHiddenUnitOutput(), eta);
        //System.out.println("deltaWeight U " + deltaWeightU.length+ "x" +deltaWeightU[0].length);
        updateWeightU(alpha, deltaWeightU);

        return output;
    }

    private void updateWeight0(double alpha, double[][] deltaWeightO) {
        double [][] weightW = getW();
        for (int k = 0; k < deltaWeightO.length; k++) {
            for (int j = 0; j < deltaWeightO[0].length; j++) {
                weightW[k][j] = weightW[k][j] + deltaWeightO[k][j] + alpha * deltaW[k][j];
                deltaW[k][j] = deltaWeightO[k][j];
            }
        }
        setW(weightW);
    }

    private double[][] getDeltaWeightO(double[] gammaO, double[] y, double eta) {
        double[][] deltaWeightO = new double[gammaO.length][y.length];
        for (int k = 0; k < gammaO.length; k++) {
            for (int j = 0; j < y.length; j++) {
                deltaWeightO[k][j] = -1.0 * eta * gammaO[k] * y[j];
            }
        }
        return deltaWeightO;
    }

    private double[] calculateGammaO(double[] target, double[] output) {
        double[] gammaO = new double[target.length];
        for (int k = 0; k < target.length; k++) {
            gammaO[k] = -1*(target[k] - output[k]) * sigmoidDerivate(output[k]);
        }
        return gammaO;
    }

    private void updateWeightY(double alpha, double[][] deltaWeightY) {
        double [][] weightV = getV();
        for (int j = 0; j < deltaWeightY.length - 1; j++) {
            for (int i = 0; i < deltaWeightY[0].length; i++) {
                weightV[j][i] = weightV[j][i] + deltaWeightY[j][i] + alpha * deltaV[j][i];
                deltaV[j][i] = deltaWeightY[j][i];
            }
        }
        setV(weightV);
    }
    
    private void updateWeightU(double alpha, double[][] deltaWeightU) {
        double [][] weightU = getU();
        for (int j = 0; j < deltaWeightU.length-1; j++) {
            for (int i = 0; i < deltaWeightU[0].length; i++) {
                weightU[j][i] = weightU[j][i] + deltaWeightU[j][i] + alpha * deltaU[j][i];
                deltaU[j][i] = deltaWeightU[j][i];
            }
        }
        setU(weightU);
    }

    private double[][] getDeltaWeightY(double[] gammaY, double[] z, double eta) {
        double[][] deltaWeightY = new double[gammaY.length][z.length];
        for (int j = 0; j < gammaY.length; j++) {
            for (int i = 0; i < z.length; i++) {
                deltaWeightY[j][i] = -1.0 * eta * gammaY[j] * z[i];
            }
        }
        return deltaWeightY;
    }

    private double[] calculateGammaY(double[] gammaO, double[] y, double[][] w) {
        double[] gammaY = new double[w[0].length];
        for (int j = 0; j < w[0].length; j++) {
            for (int k = 0; k < w.length; k++) {
                gammaY[j] += gammaO[k] * w[k][j] * sigmoidDerivate(y[j]);
            }
        }
        return gammaY;
    }
    
    private double[] calculateNonLinearVector(double[] sup) {
        double[] nonLinearVector = new double[sup.length];
        for (int i = 0; i < sup.length; i++) {
            nonLinearVector[i] = sigmoid(sup[i]);
        }
        return nonLinearVector;

    }

    private double sigmoid(double x) {
        double f = 0;
        f = 1 / (1 + Math.exp(-x));
        //System.out.println("sigmoid val" +f);
        return f;
    }

    private double sigmoidDerivate(double x) {
        double f = 0;
        f = (1 - x) * x;
        return f;
    }

   private double[] sumOfProduct(double[] augmentedVector, double[][] weights) {

        double[] sup = new double[weights.length];
        double[][] weightsT = transposeWeight(weights);
        int m = weightsT.length;
        int n = weightsT[0].length;
        if (m != augmentedVector.length) {
            System.out.println("We can not multiply this vector and matrix");
        } else {
            for (int j = 0; j < n; j++) {
                for (int i = 0; i < m; i++) {
                    sup[j] += weightsT[i][j] * augmentedVector[i];
                }
            }

        }
        return sup;
    }
   
   private double[] sumOfProductHidden(double[] augmentedVector, double[] hiddenOutput, double[][] weightsW, double[][] weightsU) {
        double[] sup = new double[weightsW.length];
        double[][] weightsT = transposeWeight(weightsW);
        int m = weightsT.length;
        int n = weightsT[0].length;
        if (m != augmentedVector.length) {
            System.out.println("We can not multiply this vector and matrix");
        } else {
            for (int j = 0; j < n; j++) {
                for (int i = 0; i < m; i++) {
                    sup[j] += weightsT[i][j] * augmentedVector[i];
                }
            }

        }
       
        //System.out.println("sup.length" +sup.length +" hidden output length" +hiddenOutput.length);
        
        double[][] weightsTU = transposeWeight(weightsU);
        //System.out.println("U.i " +weightsTU.length+" Uj"+weightsTU[0].length);
        for (int j = 0; j < sup.length; j++) {
                for (int i = 0; i < weightsTU[0].length; i++) {
                    sup[j] += weightsTU[i][j] * hiddenOutput[i];
                }
            }
        return sup;
    }


    private double[][] transposeWeight(double[][] weights) {             
        int m = weights.length;
        int n = weights[0].length;

        double[][] weightsT = new double[n][m];
        for (int j = 0; j < m; j++) {
            for (int i = 0; i < n; i++) {
                weightsT[i][j] = weights[j][i];
            }
        }
        return weightsT;
    }

    private double[] createAugmentedVector(double[] originalVector) {
        int n = originalVector.length;
        double[] augmentedVector = new double[n + 1];
        for (int i = 0; i < n; i++) {
            augmentedVector[i] = originalVector[i];
        }
        augmentedVector[n] = -1;
        return augmentedVector;
    }

    
    public double calculateSE(double[] output, double[] target) {
        double SE = 0;
        int K = output.length;
        for (int k = 0; k < K; k++) {
            double e = target[k] - output[k];
            SE += e * e;
            
        }
        SE = SE/K;        
        return SE;
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

    /**
     * @return the U
     */
    public double[][] getU() {
        return U;
    }

    /**
     * @param U the U to set
     */
    public void setU(double[][] U) {
        this.U = U;
    }

    /**
     * @return the hiddenUnitOuput
     */
    public double[] getHiddenUnitOutput() {
        return hiddenUnitOutput;
    }

    /**
     * @param hiddenUnitOuput the hiddenUnitOuput to set
     */
    public void setHiddenUnitOutput(double[] hiddenUnitOutput) {
        this.hiddenUnitOutput = hiddenUnitOutput;
    }
       
}
