package com.bushemi.backPropagation;

import com.bushemi.backPropagation.example.DataSetIOPair;
import com.bushemi.backPropagation.example.NeuralNetwork;

import java.math.BigInteger;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;


public class NnPy {
    public static final int EPOCHS = 100;
    public static final int NUM_HIDDEN = 8;
    private static final Random R = new Random();

    public static void main(String[] args) {
        double[] hidden_layer_weights = generateWeights(NUM_HIDDEN * 2);
        double[] output_layer_weights = generateWeights(NUM_HIDDEN * 2);
        double hidden_layer_bias = 0.35;
        double output_layer_bias = 0.6;
        NeuralNetwork nn = new NeuralNetwork(2,
                hidden_layer_weights,
                output_layer_weights, NUM_HIDDEN, 2,
                hidden_layer_bias, output_layer_bias);


        Map<String, Double[]> training_sets = new HashMap<>();
        Double[] training_inputs = {0.04, 0.1};
        Double[] training_outputs = {0.01, 0.99};
        training_sets.put("training_inputs", training_inputs);
        training_sets.put("training_outputs", training_outputs);

//        backPropForRandomTrainingSetNotLimitedTime(nn, training_sets);
//        backPropForRandomTrainingSet(nn, training_sets);
        backPropExample(nn, training_sets);
    }

    private static void backPropForRandomTrainingSetNotLimitedTime(NeuralNetwork nn,
                                                                   Map<String, Double[]> training_sets) {
        BigInteger i = BigInteger.ZERO;
        DataSetIOPair ioPair = new DataSetIOPair();
        while (nn.calculate_total_error(training_sets) > 0) {
            ioPair.generateNewDataset();
            nn.train(ioPair.getInput(), ioPair.getOut());
            System.out.println("i=" + i.toString()
                    + "; error=" + nn.calculate_total_error(training_sets)
                    + "; inputs= " + nn.showInput()
                    + "; outputs= " + nn.showOut());
            i = i.add(BigInteger.ONE);
        }
    }

    private static void backPropForRandomTrainingSet(NeuralNetwork nn,
                                                     Map<String, Double[]> training_sets) {
        DataSetIOPair ioPair = new DataSetIOPair();
        for (int i = 0; i < EPOCHS; i++) {
            ioPair.generateNewDataset();
            nn.train(ioPair.getInput(), ioPair.getOut());
            System.out.println("i=" + i
                    + "; error=" + nn.calculate_total_error(training_sets)
                    + "; inputs= " + nn.showInput()
                    + "; outputs= " + nn.showOut());
        }
    }

    private static void backPropExample(NeuralNetwork nn,
                                        Map<String, Double[]> training_sets) {
        final int epoch = 100;

        final Double[] inputs = {0.05, 0.1};
        final Double[] targets = {0.01, 0.99};
        for (int i = 0; i < epoch; i++) {
            nn.train(inputs, targets);
            System.out.println("i=" + i
                    + "; error=" + nn.calculate_total_error(training_sets)
                    + "; inputs= " + nn.showInput()
                    + "; outputs= " + nn.showOut());
        }
    }

    private static double[] generateWeights(int number) {
        double[] doubles = new double[number];
        for (int i = 0; i < number; i++) {
            doubles[i] = R.nextDouble();
        }
        return doubles;
    }


}


