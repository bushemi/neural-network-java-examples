package com.bushemi.backPropagation.example;


public class NeuronLayer {
    double bias;
    Neuron[] neurons;

    public NeuronLayer(int numNeurons, double bias) {
        this.bias = bias;
        neurons = new Neuron[numNeurons];
        for (int i = 0; i < numNeurons; i++) {
            // Every neuron in a layer shares the same bias
            neurons[i] = new Neuron(bias);
        }
    }


    void inspect() {
        System.out.println("neurons.length = " + neurons.length);
        for (Neuron neuron : neurons) {
            for (double weight : neuron.weights) {
                System.out.println("weight = " + weight);
            }
            System.out.println("bias = " + bias);
        }
    }

    Double[] feed_forward(Double[] inputs) {
        Double[] outputs = new Double[neurons.length];
        for (int i = 0; i < neurons.length; i++) {
            outputs[i] = neurons[i].calculateOutput(inputs);
        }
        return outputs;
    }

    double[] get_outputs() {
        double[] outputs = new double[neurons.length];
        for (int i = 0; i < neurons.length; i++) {
            outputs[i] = neurons[i].output;
        }
        return outputs;
    }


}
