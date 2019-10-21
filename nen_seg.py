# nen_seg.py
# nlp segmentation

#%matplotlib inline

import matplotlib.pyplot as plt

import nengo
import nengo_spa as spa

d = 16
vocab1 = spa.Vocabulary(16)
vocab2 = spa.Vocabulary(16)

with spa.Network() as model:
    state1 = spa.State(vocab=vocab1)
    state2 = spa.State(vocab=vocab2)
"""
d1 = 16
d2 = 16
vocab1 = spa.Vocabulary(d1)
vocab1.populate('A')
vocab2 = spa.Vocabulary(d2)
vocab2.populate('A')

with spa.Network() as model:
    state1 = spa.State(vocab=vocab1)
    state2 = spa.State(vocab=vocab2)
    spa.sym.A >> state1
    spa.reinterpret(state1) >> state2

    p = nengo.Probe(state2.output, synapse=0.03)

with nengo.Simulator(model) as sim:
    sim.run(0.5)

plt.plot(sim.trange(), spa.similarity(sim.data[p], vocab1), label='vocab1')
plt.plot(sim.trange(), spa.similarity(sim.data[p], vocab2), label='vocab2')
plt.xlabel("Time [s]")
plt.ylabel("Similarity")
plt.legend(loc='best')

plt.show()
"""
d1 = 16
d2 = 32
vocab1 = spa.Vocabulary(d1)
vocab1.populate('A')
vocab2 = spa.Vocabulary(d2)
vocab2.populate('A')

with spa.Network() as model:
    state1 = spa.State(vocab=vocab1)
    state2 = spa.State(vocab=vocab2)
    spa.sym.A >> state1
    spa.translate(state1, vocab2) >> state2

    p = nengo.Probe(state2.output, synapse=0.03)

with nengo.Simulator(model) as sim:
    sim.run(0.5)

#plt.plot(sim.trange(), spa.similarity(sim.data[p], vocab1), label='vocab1')
plt.plot(sim.trange(), spa.similarity(sim.data[p], vocab2), label='vocab2')
plt.xlabel("Time [s]")
plt.ylabel("Similarity")
plt.legend(loc='best')
plt.show()

