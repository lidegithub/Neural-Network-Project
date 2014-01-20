/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package timeseriesforecast;

import rcaller.RCaller;
import rcaller.RCode;

/**
 *
 * @author Ega
 */
public class ARIMA {

    private double[] input;
    private double[] output;

    public double[] getPredictionValueOnInputError(double[] input) {
        double [] output  = new double [input.length];
        /* Creating a RCaller */
        RCaller caller = new RCaller();
        caller.setRscriptExecutable("/usr/bin/Rscript");
        /* Creating a source code */
        RCode code = new RCode();
        code.clear();
        // add libraries needed to load data set
        code.addRCode("library(forecast)");
        
        //run forecast using auto.arima in R
        code.addDoubleArray("input", input); 
        code.addRCode("fit <- auto.arima(input)");
        code.addRCode("y <- fitted.values(fit)");
        code.addRCode("z <- y");
        code.addRCode("y_predict <- as.matrix(z)");
        code.addRCode("results<-list(output = y_predict)");
        caller.setRCode(code);
        caller.runAndReturnResult("results");
        output = caller.getParser().getAsDoubleArray("output");
        return output;
    }
}
