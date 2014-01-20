/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package timeseriesforecast;

/**
 *
 * @author Ega
 */
public class FFNN11 {

    private double[][] V;       // Weights between input and hidden layer
    private double[][] W;        // Weights between hidden and ouput layer
    private double[][] deltaV;
    private double[][] deltaW;
    private int numOfInputUnit;
    private int numOfHiddenUnit;
    private int numOfOutputUnit;
    //private int epoch;
    private double eta;
    private double alpha;

    public FFNN11(int I, int J, int K, double eta, double alpha) {
        this.numOfInputUnit = I;
        this.numOfHiddenUnit = J;
        this.numOfOutputUnit = K;
        this.V = new double[J][I + 1];
        this.deltaV = new double[J][I + 1];
        this.W = new double[K][J + 1];
        this.deltaW = new double[K][J + 1];
        //this.epoch = epoch;
        this.eta = eta;
        this.alpha = alpha;
    }

    public void initializeRandomWeight(int choice) {
        /*
         * if choice = 1 generate random number between -0.5 and 0.5 for weights initialization        
         */     
        WeightsInitialization initialWeight = new WeightsInitialization();
        int I = numOfInputUnit;
        int J = numOfHiddenUnit;
        int K = numOfOutputUnit;

        
        //use random number betweeen -0.5 and 0.5 for weights initialization
        if(choice == 1){
            //Initialize weights between input and hidden layer
            setV(initialWeight.useRandomWeight(numOfInputUnit, numOfHiddenUnit));    
            //Initialize weights between hidden and output layer
            setW(initialWeight.useRandomWeight(numOfHiddenUnit, numOfOutputUnit));        
        }

        //use Nguyen-Widrow method for weights initialization
        if(choice == 2){
            //Initialize weights between input and hidden layer
            setV(initialWeight.nguyenWidrowWeight(numOfInputUnit, numOfHiddenUnit));    
            //Initialize weights between hidden and output layer
            setW(initialWeight.nguyenWidrowWeight(numOfHiddenUnit, numOfOutputUnit));        
        }
        
        //initialization for delta
        for (int j = 0; j < J; j++) {
                for (int i = 0; i <= I; i++) {
                    deltaV[j][i] = 0.0;
                }
            }
        for (int k = 0; k < K; k++) {
            for (int j = 0; j <= J; j++) {                
                deltaW[k][j] = 0.0;
            }
        }
    }

    private void showWeights(double[][] Weights) {
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

    public double[] feedforward(double[] dataPoint) {
        //forward propagation from input to hidden layer
        double[] input = createAugmentedVector(dataPoint); //z
        double[][] v = getV();
        int m = v.length;

        int n = v[0].length;
        double[] sup = sumOfProduct(input,v);
        double[] nonLinear = calculateNonLinearVector(sup);

        //forward propagation from hidden to output layer
        double[] inputForOutput = createAugmentedVector(nonLinear); //y
        double[][] w = getW();

        double[] supOutput = sumOfProduct(inputForOutput, w);
        double[] output = calculateNonLinearVector(supOutput); //O
        return output;

    }
    
    private double[] sumOfProduct2(double[] augmentedVector, double[][] weights) {
        double[] sup = new double[weights.length];    
        int m = weights.length;
        int n = weights[0].length;
        
            for (int j = 0; j < m; j++) {
                for (int i = 0; i < n; i++) {
                    sup[j] += weights[j][i] * augmentedVector[i];
                }
            }      
        return sup;
    }
    
    public double[] gradientDescent(double[] dataPoint, double[] target, double eta, double alpha) {

        //forward propagation from input to hidden layer
        double[] input = createAugmentedVector(dataPoint); //z
        double[][] v = getV();
        double[] sup = sumOfProduct(input, v);
        double[] nonLinear = calculateNonLinearVector(sup);

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
        updateWeightY(alpha, deltaWeightY);

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
    
}
