#include "rnn.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    // Création d'un modèle de RNN
    RNN* rnn = (RNN*)malloc(sizeof(RNN));
    rnn->num_inputs = 10;
    rnn->weights = (double*)malloc(sizeof(double) * 10);
    rnn->biases = (double*)malloc(sizeof(double) * 10);

    // Formage des données d'entrée
    double* inputs = (double*)malloc(sizeof(double) * 100);
    for (int i = 0; i < 100; i++) {
        inputs[i] = i;
    }

    // Apprentissage du RNN
    train_rnn(rnn, inputs, inputs);

    // Prédiction avec le modèle de RNN
    double output = predict_rnn(rnn, inputs);
    printf("Output: %f\n", output);

    return 0;
}