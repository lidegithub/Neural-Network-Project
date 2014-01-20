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
}
