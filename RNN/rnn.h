#ifndef RNN_H
#define RNN_H

typedef struct {
    double* weights;
    double* biases;
    int num_inputs;
} RNN;

#endif  // RNN_H