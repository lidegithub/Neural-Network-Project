/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package timeseriesforecast;

/**
 *
 * @author Ega
 */
public class WeightsInitialization {
    /*
     * Here some methods to initialize weights
     */
    public double[][] useRandomWeight(int numOfSourceLayerUnit, int numOfDestinationLayerUnit) {
        /*
         * generate random number between -0.5 and 0.5 for weights initialization        
         */
        
        
        int I = numOfSourceLayerUnit;
        int J = numOfDestinationLayerUnit;
        double [][] weights = new double [J][I+1];
        for (int j = 0; j < J; j++) {
            for (int i = 0; i <= I; i++) {
                weights[j][i] = Math.random() - 0.5;               
            }
        }
        return weights;
    }
    
    public double[][] useRandomWeightForU(int numOfSourceLayerUnit, int numOfDestinationLayerUnit) {
        /*
         * generate random number between -0.5 and 0.5 for weights initialization        
         */
        
        
        int I = numOfSourceLayerUnit;
        int J = numOfDestinationLayerUnit;
        double [][] weights = new double [J][I];
        for (int j = 0; j < J; j++) {
            for (int i = 0; i <I; i++) {
                weights[j][i] = Math.random() - 0.5;               
            }
        }
        return weights;
    }
    
    public double[][] nguyenWidrowWeight(int numOfSourceLayerUnit, int numOfDestinationLayerUnit) {
        /*
         * Initialize weights based on Nguyen-Widrow Algorithm
         */
        
        
        int I = numOfSourceLayerUnit;
        int J = numOfDestinationLayerUnit;
        int n = numOfSourceLayerUnit;
        double [][] weights = new double [J][I+1];
        double [] norm = new double [numOfDestinationLayerUnit];
        double beta = 0.7*Math.pow(J, 1/n);
        System.out.println("beta" + beta);

        for (int j = 0; j < J; j++) {
            double temp = 0;
            for (int i = 0; i <= I; i++) {
                weights[j][i] = Math.random() - 0.5;
                temp += Math.pow(weights[j][i],2);
            }
            System.out.println("j"+j+" temp "+temp);
            norm[j] = Math.sqrt(temp);
        }
        
        for(int j=0; j<J; j++){
            System.out.println("norm J = "+norm[j]);
            for(int i=0; i<=I; i++){
                weights[j][i] = (beta*weights[j][i])/norm[j];
            }
        }
        return weights;
    }
    
    public double[][] nguyenWidrowWeightForU(int numOfSourceLayerUnit, int numOfDestinationLayerUnit) {
        /*
         * Initialize weights based on Nguyen-Widrow Algorithm
         */
        
        
        int I = numOfSourceLayerUnit;
        int J = numOfDestinationLayerUnit;
        int n = numOfSourceLayerUnit;
        double [][] weights = new double [J][I+1];
        double [] norm = new double [numOfDestinationLayerUnit];
        double beta = 0.7*Math.pow(J, 1/n);
        System.out.println("beta" + beta);

        for (int j = 0; j < J; j++) {
            double temp = 0;
            for (int i = 0; i <I; i++) {
                weights[j][i] = Math.random() - 0.5;
                temp += Math.pow(weights[j][i],2);
            }
            System.out.println("j"+j+" temp "+temp);
            norm[j] = Math.sqrt(temp);
        }
        
        for(int j=0; j<J; j++){
            System.out.println("norm J = "+norm[j]);
            for(int i=0; i<I; i++){
                weights[j][i] = (beta*weights[j][i])/norm[j];
            }
        }
        return weights;
    }
}
