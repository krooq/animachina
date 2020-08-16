# Project idea
Sorta like a more intense version of Conway's "Game of Life".

## Goal
Create a program that:
- Exhibits darwinian behaviour
- Can ingest information from external sources

## Notes

### Data limitations
The brain contains around 100 billion neurons, the cortex around 16 billion.
If a neuron is encoded as a single byte, we would need 100GB/16GB of information to model the brain.
This is not inconceivable, although we will probably need more than 1 byte per neuron.
We cannot load this much data into memory at once on a most systems, although with the speed of modern NVME drives it should be fine to simply read/write non-volatile storage instead.

This is way to much information to consider for a standard reinforcement learning algorithm that consumes the entire envionment to calculate its next state so we will need some process to limit the envrionment size when applying a state change function.

### Environment model

#### Selection
The envrionment must facilitate darwinian behaviour.
For this to occur, there must be some selection of elements based on fitness.
To model this we will create a fitness variable called **energy** and shall require that entities must have some **energy** to exist. We would then expect that entities better capable of obtaining **energy** shall be selected.

#### Action and Locality
Entities will be able to perform actions, which is simply a transfer of **energy** between themselves and the envrionment.
To restrict the computation to a feasible amount of data, we will also include locality variables **position** and **size**.
Through transfer of **energy** entities will be able to change their **position** and **size**.

#### Stochasticity
So we add no bias, the selection process should occur randomly.
To acomplish this we will add/remove random **energy** to the envionment at some variable rate, this is analagous to the sun shining on the surface of the earth and the surface of the earth emitting radiation to space.
We can control this variable rate to increase or decrease the available **energy** and see what happens.

#### Interpretation
We can interpret the envionment as an **energy** distribution.
Through the random **energy** fluctuations we should see the distribution evolve over time to select patterns that cluster entities together. Hopefully we will see complex patterns emerge!


### Neuron model
We hope that the envionment will create complex patterns, once we have this we want those patterns to behave like a neuron does.

Neurons:
- Transmit information by sending electrical impulses as a **spike** in electrical potential
- Encode information as a probability to **spike** given some input signals as a function of time

If we consider a simple neuron model that holds some state and can send a spike impulse we see that it is not so different from a collection of entities defined above.
Hopefully we can adjust the environment parameters sufficiently to see this spiking behaviour from clusters of entities.

# Spiking Model by Izhikevich (2003)

v' = 0.04v^2 + 5v + 140 - u + I           (1)
u' = a(bv - u)                            (2)
if v >= 30 then { v = c, u = u + d }      (3)

Here variable v represents the membrane potential of the neuron 
and u represents a membrane recovery variable.
After the spike reaches its apex 30mV, the membrane voltage and the 
recovery variable are reset according to the (3). 
If v skips over 30, then it first is reset to 30, and then to c so 
that all spikes have equal magnitudes.
The quadratic part is chosen so that has mV scale and the time has ms scale. 
It takes only 13 floating point operation to simulate 1ms of the model,
so it is quite efficient in large-scale simulations of cortical networks.
When (a,b,c,d) = (0.2, 2, -56, -16) and I = -99and, the model has chaotic spiking activity,
though the integration time step should be small to achieve adequate numerical precision.
We stress that 30mV in (3) is not a threshold, but the peak of the spike.
The threshold value of the model neuron is between –70 and -50, and it is dynamic, as in biological neurons.