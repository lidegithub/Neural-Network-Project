/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package timeseriesforecast;

import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Random;
import rcaller.Globals;
import rcaller.RCaller;
import rcaller.RCode;

/**
 *
 * @author Ega
 */
public class RToJava {

    public void resultTest() {
        try {

            /*
             * Creating Java's random number generator
             */
            Random random = new Random();

            /*
             * Creating RCaller
             */
            RCaller caller = new RCaller();
            RCode code = new RCode();
            /*
             * Full path of the Rscript. Rscript is an executable file shipped with R.
             * It is something like C:\\Program File\\R\\bin.... in Windows
             */
            caller.setRscriptExecutable("/usr/local/bin/Rscript");

            /*
             *  We are creating a random data from a normal distribution
             * with zero mean and unit variance with size of 100
             */
            double[] data = new double[100];

            for (int i = 0; i < data.length; i++) {
                data[i] = random.nextGaussian();
            }

            /*
             * We are transferring the double array to R
             */
            code.addDoubleArray("x", data);

            /*
             * Adding R Code
             */
            code.addRCode("my.mean<-mean(x)");
            code.addRCode("my.var<-var(x)");
            code.addRCode("my.sd<-sd(x)");
            code.addRCode("my.min<-min(x)");
            code.addRCode("my.max<-max(x)");
            code.addRCode("my.standardized<-scale(x)");

            /*
             * Combining all of them in a single list() object
             */
            code.addRCode(
                    "my.all<-list(mean=my.mean, variance=my.var, sd=my.sd, min=my.min, max=my.max, std=my.standardized)");

            /*
             * We want to handle the list 'my.all'
             */
            caller.setRCode(code);
            caller.runAndReturnResult("my.all");

            double[] results;

            /*
             * Retrieving the 'mean' element of list 'my.all'
             */
            results = caller.getParser().getAsDoubleArray("mean");
            System.out.println("Mean is " + results[0]);

            /*
             * Retrieving the 'variance' element of list 'my.all'
             */
            results = caller.getParser().getAsDoubleArray("variance");
            System.out.println("Variance is " + results[0]);

            /*
             * Retrieving the 'sd' element of list 'my.all'
             */
            results = caller.getParser().getAsDoubleArray("sd");
            System.out.println("Standard deviation is " + results[0]);

            /*
             * Retrieving the 'min' element of list 'my.all'
             */
            results = caller.getParser().getAsDoubleArray("min");
            System.out.println("Minimum is " + results[0]);

            /*
             * Retrieving the 'max' element of list 'my.all'
             */
            results = caller.getParser().getAsDoubleArray("max");
            System.out.println("Maximum is " + results[0]);

            /*
             * Retrieving the 'std' element of list 'my.all'
             */
            results = caller.getParser().getAsDoubleArray("std");

            /*
             * Now we are retrieving the standardized form of vector x
             */
            System.out.println("Standardized x is ");

            for (int i = 0; i < results.length; i++) {
                System.out.print(results[i] + ", ");
            }
        } catch (Exception e) {
            System.out.println(e.toString());
        }
    }

    public void plotTest() {
        try {
            RCaller caller = new RCaller();
            caller.setRscriptExecutable("/usr/local/bin/Rscript");
            caller.cleanRCode();

            double[] numbers = new double[]{1, 4, 3, 5, 6, 10};

            caller.addDoubleArray("x", numbers);
            File file = caller.startPlot();
            System.out.println(file);
            caller.addRCode("plot.ts(x)");
            caller.endPlot();
            caller.runOnly();
            caller.showPlot(file);
        } catch (Exception e) {
            System.out.println(e.toString());
        }
    }

    public void readFromFile() {
        /* Creating a RCaller */
        RCaller caller = new RCaller();
        caller.setRscriptExecutable("/usr/local/bin/Rscript");

        /* Creating a source code */
        RCode code = new RCode();
        code.clear();

        /* Creating an external data file 
         * Suppose that the data file is like
         * X Y Z
         * 1 2 3
         * 4 5 6
         * 7 8 9
         * 10 11 12
         */
        File f = null;
        try {
            f = File.createTempFile("rcallerexmp", "");
            FileWriter writer = new FileWriter(f);
            PrintWriter pwriter = new PrintWriter(writer);
            pwriter.println("X Y Z");
            pwriter.println("1 2 3");
            pwriter.println("4 5 6");
            pwriter.println("7 8 9");
            pwriter.println("10 11 12");
            pwriter.flush();
            pwriter.close();
        } catch (Exception e) {
            System.out.println("Error while writing to external data file");
        }

        /* Now, writing some R Code */
        code.addRCode("data<-read.table(\"" + f.getAbsoluteFile() + "\", header=TRUE)");

        /* Running the Code */
        caller.setRCode(code);
        caller.runAndReturnResult("data");

        /* Getting Results */
        double[] Z = caller.getParser().getAsDoubleArray("Z");

        /* Printing Z */
        for (int i = 0; i < Z.length; i++) {
            System.out.println(Z[i]);
        }

    }

    public double [] readDataMarket() {
        double[] results = null;
        try {
            /* Creating a RCaller */
            RCaller caller = new RCaller();
            caller.setRscriptExecutable("/usr/local/bin/Rscript");

            /* Creating a source code */
            RCode code = new RCode();
            code.clear();
            code.addRCode("library(zoo)");
            code.addRCode("library(timeSeries)");
            code.addRCode("library(rdatamarket)");
            code.addRCode("dminit(\"be0123ef782e49348a7ed53c2444c08c\")");
            code.addRCode("a <- dmlist(\"22vh\")");
            code.addRCode("results<-list(mydata = a[,2])");
            caller.setRCode(code);
            System.out.println("script exe" + caller.getRscriptExecutable());
            //caller.runAndReturnResult("results");
            
            caller.runAndReturnResult("results");

            
            results = caller.getParser().getAsDoubleArray("mydata");
            for (int i = 0; i < results.length; i++) {
                System.out.println("s " + results[i]);
            }
            
        } catch (Exception ex) {
            System.out.println(ex.getMessage());
        }
        return results;
    }

    public void testTimeSeries() {
        try {
            /* Creating a RCaller */
            RCaller caller = new RCaller();
            caller.setRscriptExecutable("/usr/local/bin/Rscript");
            caller.cleanRCode();

            /* Creating a source code */
            RCode code = new RCode();
            code.clear();

            code.addRCode("library(timeSeries)");
            code.addRCode("data <- matrix(1:24, ncol = 2)");


            code.addRCode("s <- timeSeries(data, timeCalendar())");
            code.addRCode("results<-list(mydata = s[,1], mydata2 = s[,2])");

            caller.setRCode(code);
            System.out.println("script exe" + caller.getRscriptExecutable());
            caller.runAndReturnResult("results");

            double[] results;
            results = caller.getParser().getAsDoubleArray("mydata2");
            for (int i = 0; i < results.length; i++) {
                System.out.println("s " + results[i]);
            }


        } catch (Exception ex) {
            System.out.println(ex.getMessage());
        }

    }

    public static void main(String[] args) {
        RToJava rtj = new RToJava();
        //rtj.resultTest();
        //rtj.plotTest();
        rtj.readDataMarket();
    }
}
