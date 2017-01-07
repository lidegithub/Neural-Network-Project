/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package timeseriesforecast;

import java.io.File;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;
import rcaller.RCaller;
import rcaller.RCode;

/**
 *
 * @author Ega
 */
public class ForecastNNHyperbolic {

    
    public ForecastNNHyperbolic() {
    }

    public double[] getDataSet() {
        double[] results = null;
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

            //get data set
            code.addRCode("dminit(\"be0123ef782e49348a7ed53c2444c08c\")");
            code.addRCode("dataSet <- dmlist(\"22nv\")");
            code.addRCode("results<-list(nnData = dataSet[,2], generalData = dataSet)");
            caller.setRCode(code);
            System.out.println("script exe" + caller.getRscriptExecutable());
            caller.runAndReturnResult("results");
            results = caller.getParser().getAsDoubleArray("nnData");

        } catch (Exception ex) {
            System.out.println(ex.getMessage());
        }
        return results;
    }
        
    public void trainNN() {
        TimeSeriesNNHyperbolic tsnn = new TimeSeriesNNHyperbolic();
        double [] dataSet = getDataSet();
        tsnn.setMinMax(dataSet);
        double[] results = tsnn.normalizeData(dataSet);
        System.out.println("min value"+tsnn.getMinValue());
        System.out.println("max value"+tsnn.getMaxValue());

        /*
         * System.out.println("Data size = "+results.length);
         for (int i = 0; i < results.length; i++) {
         System.out.println("s " + results[i]);
         }
         */
        double[] RMSE = new double[101];
        RMSE = tsnn.TrainingNN(1, results, 12, 12, 1, 0.25, 0.25, 500, 5);
        try {
            RCaller caller = new RCaller();
            caller.setRscriptExecutable("/usr/local/bin/Rscript");
            caller.cleanRCode();
            File file;
            int[] arr = new int[1];
            arr[0] = 1;
            file = caller.startPlot();
            caller.addDoubleArray("RMSE", RMSE);
            caller.addIntArray("arr", arr);
            caller.addRCode("a = arr[1]");
            caller.addRCode("plot.ts(RMSE, main=a)");
            caller.endPlot();
            caller.runOnly();
            caller.showPlot(file);
        } catch (IOException ex) {
            Logger.getLogger(FFNN.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    public static void main(String[] args) {
        ForecastNNHyperbolic fnn = new ForecastNNHyperbolic();
        fnn.trainNN();
    }

}
