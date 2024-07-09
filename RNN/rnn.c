#include "rnn.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void update_hidden_state(RNN* rnn, double* inputs, double* hidden_state) {
    // Calcul de l'état interne à partir des poids et des biases
    for (int i = 0; i < rnn->num_inputs; i++) {
        hidden_state[i] += rnn->weights[i] * inputs[i];
    }
    // Mise à jour de l'état interne en fonction des poids et des biases
    for (int i = 0; i < rnn->num_inputs; i++) {
        hidden_state[i] += rnn->biases[i];
    }
}

void train_rnn(RNN* rnn, double* inputs, double* targets) {
    // Mise à jour de l'état interne (hidden state)
    double* hidden_state = (double*)malloc(sizeof(double) * rnn->num_inputs);
    update_hidden_state(rnn, inputs, hidden_state);

    // Calcul de la sortie du RNN
    double output = 0;
    for (int i = 0; i < rnn->num_inputs; i++) {
        output += hidden_state[i] * rnn->weights[i];
    }

    // Mise à jour des poids et des biases en fonction de l'erreur
    double error = 0;
    for (int i = 0; i < rnn->num_inputs; i++) {
        error += (output - targets[i]) * (hidden_state[i] + 1);
    }
    // Mise à jour des poids et des biases en fonction de l'erreur
    for (int i = 0; i < rnn->num_inputs; i++) {
        rnn->weights[i] -= 0.01 * error;
        rnn->biases[i] -= 0.01 * error;
    }
}

double predict_rnn(RNN* rnn, double* inputs) {
    // Mise à jour de l'état interne (hidden state)
    double* hidden_state = (double*)malloc(sizeof(double) * rnn->num_inputs);
    update_hidden_state(rnn, inputs, hidden_state);

    // Calcul de la sortie du RNN
    double output = 0;
    for (int i = 0; i < rnn->num_inputs; i++) {
        output += hidden_state[i] * rnn->weights[i];
    }

    return output;
}