/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package timeseriesforecast;

/**
 *
 * @author Ega
 */
public class Choice {    
    private int numOfInputUnit;
    private int numOfHiddenUnit;
    private int numOfOutputUnit;
    private int epoch;
    private double eta;
    private double alpha;
    
    private double[][] V ;       // Weights between input and hidden layer
    private double[][] W;        // Weights between hidden and ouput layer


    /**
     * @return the numOfInputUnit
     */
    public int getNumOfInputUnit() {
        return numOfInputUnit;
    }

    /**
     * @param numOfInputUnit the numOfInputUnit to set
     */
    public void setNumOfInputUnit(int numOfInputUnit) {
        this.numOfInputUnit = numOfInputUnit;
    }

    /**
     * @return the numOfHiddenUnit
     */
    public int getNumOfHiddenUnit() {
        return numOfHiddenUnit;
    }

    /**
     * @param numOfHiddenUnit the numOfHiddenUnit to set
     */
    public void setNumOfHiddenUnit(int numOfHiddenUnit) {
        this.numOfHiddenUnit = numOfHiddenUnit;
    }

    /**
     * @return the numOfOutputUnit
     */
    public int getNumOfOutputUnit() {
        return numOfOutputUnit;
    }

    /**
     * @param numOfOutputUnit the numOfOutputUnit to set
     */
    public void setNumOfOutputUnit(int numOfOutputUnit) {
        this.numOfOutputUnit = numOfOutputUnit;
    }

    /**
     * @return the epoch
     */
    public int getEpoch() {
        return epoch;
    }

    /**
     * @param epoch the epoch to set
     */
    public void setEpoch(int epoch) {
        this.epoch = epoch;
    }

    /**
     * @return the eta
     */
    public double getEta() {
        return eta;
    }

    /**
     * @param eta the eta to set
     */
    public void setEta(double eta) {
        this.eta = eta;
    }

    /**
     * @return the alpha
     */
    public double getAlpha() {
        return alpha;
    }

    /**
     * @param alpha the alpha to set
     */
    public void setAlpha(double alpha) {
        this.alpha = alpha;
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
