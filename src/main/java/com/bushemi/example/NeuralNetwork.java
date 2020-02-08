package com.bushemi.example;


import java.util.Map;
import java.util.Random;

public class NeuralNetwork {
    private static final double LEARNING_RATE = 0.5;
    private static final Random RANDOM = new Random();
    private Double[] training_inputs;
    private Double[] training_outputs;
    private int num_inputs;
    private NeuronLayer hidden_layer;
    private NeuronLayer output_layer;

    public NeuralNetwork(int num_inputs,
                         double[] hidden_layer_weights,
                         double[] output_layer_weights,
                         int num_hidden, int num_outputs,
                         double hidden_layer_bias, double output_layer_bias) {
        this.num_inputs = num_inputs;

        this.hidden_layer = new NeuronLayer(num_hidden, hidden_layer_bias);
        this.output_layer = new NeuronLayer(num_outputs, output_layer_bias);

        init_weights_from_inputs_to_hidden_layer_neurons(hidden_layer_weights);
        init_weights_from_hidden_layer_neurons_to_output_layer_neurons(output_layer_weights);
    }

    public String showInput() {
        String s = "";
        for (Double training_input : training_inputs) {
            s += ("in=" + training_input + " ");
        }
        s += "";
        return s;
    }

    public String showOut() {
        String s = "";
        for (Double training_output : training_outputs) {
            s += ("out=" + training_output + " ");
        }
        s += "";
        return s;
    }

    public double calculate_total_error(Map<String, Double[]> training_sets) {
        double totalError = 0;
        for (int i = 0; i < training_sets.get("training_inputs").length; i++) {
            Double[] trainingInputs = training_sets.get("training_inputs");
            Double[] trainingOutputs = training_sets.get("training_outputs");
            feed_forward(trainingInputs);
            for (int j = 0; j < trainingOutputs.length; j++) {
                totalError += output_layer.neurons[j].calculate_error(trainingOutputs[j]);
            }
        }
        return totalError;
    }


    Double[] feed_forward(Double[] inputs) {
        Double[] hidden_layer_outputs = hidden_layer.feed_forward(inputs);
        return output_layer.feed_forward(hidden_layer_outputs);
    }


    void inspect() {
        System.out.println("------");
        System.out.print("* Inputs: =" + num_inputs);
        System.out.println("------");
        hidden_layer.inspect();
        System.out.println("------");
        output_layer.inspect();
        System.out.println("------");
    }


    void init_weights_from_hidden_layer_neurons_to_output_layer_neurons(double[] output_layer_weights) {
        int weight_num = 0;
        for (int i = 0; i < output_layer.neurons.length; i++) {
            for (int j = 0; j < hidden_layer.neurons.length; j++) {
                if (output_layer_weights.length == 0) {
                    output_layer.neurons[i].weights.add(RANDOM.nextDouble());
                } else {
                    output_layer.neurons[i].weights.add(output_layer_weights[weight_num]);
                }
                weight_num += 1;
            }
        }
    }

    void init_weights_from_inputs_to_hidden_layer_neurons(double[] hidden_layer_weights) {
        int weight_num = 0;
        for (int i = 0; i < hidden_layer.neurons.length; i++) {
            for (int j = 0; j < num_inputs; j++) {
                if (hidden_layer_weights.length == 0) {
                    hidden_layer.neurons[i].weights.add(RANDOM.nextDouble());
                } else {
                    hidden_layer.neurons[i].weights.add(hidden_layer_weights[weight_num]);
                }
                weight_num += 1;
            }
        }
    }

    //     Uses online learning, ie updating the weights after each training case
    public void train(Double[] training_inputs, Double[] training_outputs) {
        this.training_inputs = training_inputs;
        this.training_outputs = training_outputs;
        feed_forward(training_inputs);

//                 1. Output neuron deltas
        double[] pd_errors_wrt_output_neuron_total_net_input = new double[output_layer.neurons.length];
        for (int i = 0; i < output_layer.neurons.length; i++) {

//                 ∂E/∂zⱼ
            pd_errors_wrt_output_neuron_total_net_input[i] = output_layer.neurons[i].calculate_pd_error_wrt_total_net_input(training_outputs[i]);
        }

//         2. Hidden neuron deltas
        double[] pd_errors_wrt_hidden_neuron_total_net_input = new double[hidden_layer.neurons.length];
        for (int i = 0; i < hidden_layer.neurons.length; i++) {
//                 We need to calculate the derivative of the error with respect to the output of each hidden layer neuron
//             dE/dyⱼ = Σ ∂E/∂zⱼ * ∂z/∂yⱼ = Σ ∂E/∂zⱼ * wᵢⱼ

            double d_error_wrt_hidden_neuron_output = 0;
            for (int j = 0; j < output_layer.neurons.length; j++) {
                d_error_wrt_hidden_neuron_output += pd_errors_wrt_output_neuron_total_net_input[j] * output_layer.neurons[j].weights.get(i);
            }
            //                 ∂E/∂zⱼ = dE/dyⱼ * ∂zⱼ/∂
            pd_errors_wrt_hidden_neuron_total_net_input[i] = d_error_wrt_hidden_neuron_output * hidden_layer.neurons[i].calculate_pd_total_net_input_wrt_input();
        }

//        3. Update output neuron weights
        for (int i = 0; i < output_layer.neurons.length; i++) {
            for (int j = 0; j < output_layer.neurons[i].weights.size(); j++) {

//                ∂Eⱼ/∂wᵢⱼ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢⱼ
                double pd_error_wrt_weight = pd_errors_wrt_output_neuron_total_net_input[i] * output_layer.neurons[i].calculate_pd_total_net_input_wrt_weight(j);

                //                 Δw = α * ∂Eⱼ/∂wᵢ
                double newOutWeight = output_layer.neurons[i].weights.get(j) - LEARNING_RATE * pd_error_wrt_weight;
                output_layer.neurons[i].weights.set(j, newOutWeight);
            }
        }

//         4. Update hidden neuron weights
        for (int i = 0; i < hidden_layer.neurons.length; i++) {
            for (int j = 0; j < hidden_layer.neurons[i].weights.size(); j++) {

//                 ∂Eⱼ/∂wᵢ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢ
                double pd_error_wrt_weight = pd_errors_wrt_hidden_neuron_total_net_input[i] * hidden_layer.neurons[i].calculate_pd_total_net_input_wrt_weight(j);

//                 Δw = α * ∂Eⱼ/∂wᵢ
                hidden_layer.neurons[i].weights.set(j, (LEARNING_RATE * pd_error_wrt_weight));
            }
        }
    }

}


