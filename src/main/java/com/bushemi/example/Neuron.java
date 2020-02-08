package com.bushemi.example;

import java.util.ArrayList;
import java.util.List;

class Neuron {
    public double bias;
    public double output;
    public List<Double> weights = new ArrayList<>();
    public Double[] inputs;

    public Neuron(double bias) {
        this.bias = bias;
    }

    public double calculateOutput(Double[] inputs) {
        this.inputs = inputs;
        this.output = this.squash(this.calculateTotalNetInput());
        return output;
    }

    public double calculateTotalNetInput() {
        double total = 0;
        for (int i = 0; i < inputs.length; i++) {
            total += inputs[i] * weights.get(i);
        }
        return total + bias;
    }

    //        Apply the logistic function to squash the output of the neuron
//        The result is sometimes referred to as 'net' [2] or 'net' [1]
    public double squash(double totalNetInput) {
        return 1 / (1 + Math.exp(-totalNetInput));
    }

    //    Determine how much the neuron's total input has to change to move closer to the expected output
//
//                 Now that we have the partial derivative of the error with respect to the output (∂E/∂yⱼ) and
//     the derivative of the output with respect to the total net input (dyⱼ/dzⱼ) we can calculate
//     the partial derivative of the error with respect to the total net input.
//     This value is also known as the delta (δ) [1]
//                 δ = ∂E/∂zⱼ = ∂E/∂yⱼ * dyⱼ/dzⱼ
    public double calculate_pd_error_wrt_total_net_input(double targetOutput) {
        return calculate_pd_error_wrt_output(targetOutput) * calculate_pd_total_net_input_wrt_input();
    }

    //     The error for each neuron is calculated by the Mean Square Error method:
    public double calculate_error(double targetOutput) {
//        System.out.println("output = " + output);
//        if (targetOutput >= output) {
//            return targetOutput;
//        } else {
//            return output;
//        }
//        return Double.compare(targetOutput, output);
        return 0.5 * Math.pow((targetOutput - output), 2);
    }

    //             The partial derivate of the error with respect to actual output then is calculated by:
//                 = 2 * 0.5 * (target output - actual output) ^ (2 - 1) * -1
//                 = -(target output - actual output)
//
//                 The Wikipedia article on backpropagation [1] simplifies to the following, but most other learning material does not [2]
//                 = actual output - target output
//
//             Alternative, you can use (target - output), but then need to add it during backpropagation [3]
//
//                 Note that the actual output of the output neuron is often written as yⱼ and target output as tⱼ so:
//                 = ∂E/∂yⱼ = -(tⱼ - yⱼ)
    public double calculate_pd_error_wrt_output(double targetOutput) {
//        return Double.compare(targetOutput, output);
        return -(targetOutput - output);
    }

    //     The total net input into the neuron is squashed using logistic function to calculate the neuron's output:
//                 yⱼ = φ = 1 / (1 + e^(-zⱼ))
//                 Note that where ⱼ represents the output of the neurons in whatever layer we're looking at and ᵢ represents the layer below it
//
//                 The derivative (not partial derivative since there is only one variable) of the output then is:
//                 dyⱼ/dzⱼ = yⱼ * (1 - yⱼ)
    public double calculate_pd_total_net_input_wrt_input() {
        return output * (1 - output);
    }


    //     The total net input is the weighted sum of all the inputs to the neuron and their respective weights:
//                 = zⱼ = netⱼ = x₁w₁ + x₂w₂ ...
//
//                 The partial derivative of the total net input with respective to a given weight (with everything else held constant) then is:
//                 = ∂zⱼ/∂wᵢ = some constant + 1 * xᵢw₁^(1-0) + some constant ... = xᵢ
    public double calculate_pd_total_net_input_wrt_weight(int index) {
        return inputs[index];
    }


}
