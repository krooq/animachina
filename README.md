# Animachina

From the latin anima + machina = life machine... I'm so edgy...

Anyway this is just a place to practice my AI/ML and Python.

## Projects
### Spiking Neuron based AI system

#### Architecture
The end-to-end architecture for data flow is:
- Environment
- Sensors
- Routers
- Neurons
- Routers
- Actuators
- Environment

#### Neuron Model
| parameter         | dtype | shape | variability | notes                                                                                           |
|-------------------|-------|-------|-------------|-------------------------------------------------------------------------------------------------|
| position          | f32   | (2,)  | static      | The position of the neuron, this value uniquely identifies the neuron                           |
| target            | f32   | (2,)  | static      | The *position* of the target neuron to receive input when this neuron **activates**             |
| inputs            | f32   | (n,2) | dynamic     | The *positions* of neurons that transfer some *potential* to this neuron when they **activate** |
| sensitivity       | f32   | (n,)  | dynamic     | The multipliers applied to incoming *inputs* when each input **activates**                      |
| potential         | f32   | ()    | dynamic     | The current excitation of the neuron, if this exceeds *threshold* the neuron will **activate**  |
| threshold         | f32   | ()    | static      | The threshold until activation, if *potential* exceeds this the neuron will **activate**        |
| rest-potential    | f32   | ()    | static      | The value the *potential* of the neuron starts at                                               |
| sensitivity delta | f32   | ()    | static      | How much the *sensitivity* changes w.r.t. **input activation** and this neurons **activation**  |
| transfer delta    | f32   | ()    | static      | How much *potential* is transferred to the *target* neuron when this neuron **activates**       |

#### Signals

##### Data signals
Data signals are the interface type between hardware drivers and software.
There are many formats but at some level they are all just streams of arrays of data.
Examples include video data, audio data, instructions for mechanical operations etc.

##### Spike signals
Spike signals are the interface type between regular software and the AI system software.
These signals have a simple format, usually just a stream of a single number to indicate the spike level.

#### Spike Transducers
Spiking Neurons rely on input signals that vary in the time domain.
Spike transducers convert *data signals* into *spike signals* that the neuron can process.
They also perform the reverse transformation from *spike signals* to *data signals* that regular software can process.

##### Sensors
Sensors are the inputs to the artificial intelligence system.

Sensors take some data and increases the *potential* of some neurons in a localized region.
The data can be encoded by any method that maps input data to neuron potential.
e.g. For a text processor the letter "a" can be mapped to a single neuron, 
if you wanted you could also map "ab" to a single neuron but then you would need to repeat this for "ac" and so on.
For an visual processor, you would need to define a sensor size and downscale inputs to match, you would also need a neuron for each color RGB.
The alpha channel wouldn't need to be processed by the neuron but it may need to be blended with other images if processing raw data, it can be tricky.

##### Actuators
Actuators are the outputs of the artificial intelligence system.

Actuators take the *potentials* of some neurons in a localized region and turns them into some data.
Again, any mapping from a single neuron to single piece of output data will work, just follow the reverse process of the sensors.

#### Routers
Ok so we receive a signal from a sensor, it is propogated through the neurons which store the pattern, then somewhere it emerges from another set of neurons, and then into an actuator to do a thing.
But which neurons do we wire up our actuator to?
And what if we want to do many things using different actuators triggered by a single input signal?
For these functions we need something that tells the neurons where to send their signals.
In the brain this the routing of signals is hardwired (more or less), each section of the brain performs a different task and neurons are wired to send signals to the correct sections.
We will need to do some wiring of our own to get our signals to the right location.

##### Reward Pathway
How can we get a thing to do a thing? This ability doesn't just emerge out of the above neuron model.
One of the major features that defines intelligence is the ability to integrate information from multiple sources and react to it.
This creates a feedback loop, but to make any kind of change to the model we need to have some sort of policy to adapt to the feedback.
A policy is typically fixed in most ML algorithms as they are designed to achieve one single task, out model however will need to adapt its policy.
In the brain it is thought that there is a dedicated systems e.g. "dopamine pathway" that regulates model changes.

#### Investigation Topics
##### "Always-On" input signal
How does a spiking neuron handle signals which are "always-on" e.g. staring at an image?
Does the transducer handle *debouncing* of the signal so that only changes are propogated? 
If so, how long does it work for us, since we take some time to ingest visual information?
That being said we only need to "hear things once" and "see things once" to ingest them...
TODO: research how visual neurons work for "always-on" information