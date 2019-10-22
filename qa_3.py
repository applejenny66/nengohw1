#qa_2.py


import matplotlib.pyplot as plt
#%matplotlib inline

import nengo
import nengo_spa as spa

# Number of dimensions for the Semantic Pointers
dimensions = 32

model = spa.Network(label="question answering")

def question_input(t):
    if (t // 0.2) % 5 == 0:
        return 'WHEN'
    elif (t // 0.2) % 5 == 1: 
        return 'WHO'
    elif (t // 0.2) % 5 == 2:
        return 'WHAT'
    elif (t // 0.2) % 5 == 3:
        return 'WHOM' 
    else:
        return 'HOWLONG'

def corespond_input(t):
    if (t // 0.2) % 5 == 0:
        return 'JANUARY2016'
    elif (t // 0.2) % 5 == 1: 
        return 'BNPPARIBAS'
    elif (t // 0.2) % 5 == 2:
        return 'EXCELLENCEPROGRAM'
    elif (t // 0.2) % 5 == 3:
        return 'GRADUATES' 
    else:
        return 'MONTHS'

def cue_input(t):
    sequence = ['0', 'WHEN', 'JANUARY2016', '0', 'WHO', 'BNPPARIBAS', '0', \
        'WHAT',  'EXCELLENCEPROGRAM',  '0', 'WHOM' ,  'GRADUATES' , '0', \
        'HOWLONG', 'MONTHS']
    idx = int((t // (1. / len(sequence))) % len(sequence))
    return sequence[idx]

with model:
    color_in = spa.Transcode(question_input, output_vocab=dimensions)
    shape_in = spa.Transcode(corespond_input, output_vocab=dimensions)
    cue = spa.Transcode(cue_input, output_vocab=dimensions)

with model:
    conv = spa.State(dimensions)
    out = spa.State(dimensions)

    # Connect the buffers
    color_in * shape_in >> conv
    conv * ~cue >> out

with model:
    model.config[nengo.Probe].synapse = nengo.Lowpass(0.03)
    p_color_in = nengo.Probe(color_in.output)
    p_shape_in = nengo.Probe(shape_in.output)
    p_cue = nengo.Probe(cue.output)
    p_conv = nengo.Probe(conv.output)
    p_out = nengo.Probe(out.output)

with nengo.Simulator(model) as sim:
    sim.run(2.5)

plt.figure(figsize=(10, 10))
vocab = model.vocabs[dimensions]
"""
plt.subplot(3, 1, 1)
plt.plot(sim.trange(), spa.similarity(sim.data[p_color_in], vocab))
plt.legend(vocab.keys(), fontsize='x-small')
plt.ylabel("question")

plt.subplot(3, 1, 2)
plt.plot(sim.trange(), spa.similarity(sim.data[p_shape_in], vocab))
plt.legend(vocab.keys(), fontsize='x-small')
plt.ylabel("corespond")

plt.subplot(3, 1, 3)
plt.plot(sim.trange(), spa.similarity(sim.data[p_cue], vocab))
plt.legend(vocab.keys(), fontsize='x-small')
plt.ylabel("cue")

"""
plt.subplot(2, 1, 1)
for pointer in ['WHEN * JANUARY2016', 'WHO * BNPPARIBAS',  'WHAT * EXCELLENCEPROGRAM', \
    'WHOM * GRADUATES', 'HOWLONG * MONTHS']:
    plt.plot(sim.trange(), vocab.parse(pointer).dot(sim.data[p_conv].T), label=pointer)
plt.legend(fontsize='x-small')
plt.ylabel("convolved")

plt.subplot(2, 1, 2)
plt.plot(sim.trange(), spa.similarity(sim.data[p_out], vocab))
plt.legend(vocab.keys(), fontsize='x-small')
plt.ylabel("output")

plt.xlabel("time [s]")

plt.savefig('test3.jpg')
plt.show()


