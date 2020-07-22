# Animachina

From the latin anima + machina = life machine... I'm so edgy...

Anyway this is just a place to practice my AI/ML and Python.

## Projects
### Spiking neuron model
| parameter         | dtype | shape | variability | notes                                                                                           |
|-------------------|-------|-------|-------------|-------------------------------------------------------------------------------------------------|
| position          | f32   | (2,)  | static      | The position of the neuron, this value uniquely identifies the neuron                           |
| target            | f32   | (2,)  | static      | The *position* of the target neuron to receive input when this neuron **activates**             |
| inputs            | f32   | (n,2) | dynamic     | The *positions* of neurons that transfer some *potential* to this neuron when they **activate** |
| sensitivity       | f32   | (n,)  | dynamic     | The multipliers applied to incoming *inputs* when each input **activates**                      |
| potential         | f32   | ()    | dynamic     | The current excitation of the neuron, if this exceeds *threshold* the neuron will **activate**  |
| threshold         | f32   | ()    | static      | The threshold until activation, if *potential* exceeds this the neuron will **activate**        |
| rest-potential    | f32   | ()    | static      | The value the *potential* of the neuron starts at                                               |
| sensitivity delta | f32   | ()    | static      | How much the *sensitivity* changes w.r.t. *input activation* and this neurons *activation*      |
| transfer delta    | f32   | ()    | static      | How much potential is transferred to the *target* neuron when this neuron *activates*           |
