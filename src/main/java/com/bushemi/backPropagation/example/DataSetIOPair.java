package com.bushemi.backPropagation.example;

import java.util.Random;

public class DataSetIOPair {
    private Double[] input;
    private Double[] out;
    private static final Random RANDOM = new Random();

    public DataSetIOPair() {
        input = new Double[2];
        out = new Double[2];
    }

    public void generateNewDataset() {
        input[0] = RANDOM.nextDouble();
        input[1] = RANDOM.nextDouble();
        if (input[0] > input[1]) {
            out[0] = 0.99;
            out[1] = 0.01;
        } else {
            out[1] = 0.99;
            out[0] = 0.01;
        }
    }

    public Double[] getInput() {
        return input;
    }

    public Double[] getOut() {
        return out;
    }
}
