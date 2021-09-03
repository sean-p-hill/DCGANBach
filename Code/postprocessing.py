import sys
import os
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import matplotlib
matplotlib.rcParams['animation.embed_limit'] = 2**128
matplotlib.rcParams['animation.ffmpeg_path'] = r'../Data/ffmpeg/bin/ffmpeg.exe'
import numpy as np
import random
from functools import reduce
from IPython.display import HTML

from midi2audio import FluidSynth

from PIL import Image
from music21 import instrument, note, chord, stream, pitch, tempo

import torch
import torchvision

# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
# print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)


lowerBoundNote = 0
# For debugging usage only...
def nn(n):
    # MIDI number to note names for iterable
    return [str(pitch.Pitch(i)) for i in n] if isinstance(n,list) else str(pitch.Pitch(n))

def column2notes(column,min_pitch,max_pitch):
    notes = []
    for i in range(len(column)):
        if column[i] > 255/2:
            notes.append((128-min_pitch)-i+lowerBoundNote)
    return notes

resolution = 0.5
def updateNotes(newNotes,prevNotes): 
    res = {} 
    for note in newNotes:
        if note in prevNotes:
            res[note] = prevNotes[note] + resolution
        else:
            res[note] = resolution
    return res

def image2midi(image_path,min_pitch,max_pitch, toaudio=True, play=False):
    '''
    Convert images to MIDi files and then subsequently to .wav files
    '''

    with Image.open(image_path) as image:
        im_arr = np.fromstring(image.convert('L').tobytes(), dtype=np.uint8)
        
        try:
            im_arr = im_arr.reshape((image.size[1], image.size[0]))
        except:
            im_arr = im_arr.reshape((image.size[1], image.size[0],3))
            im_arr = np.dot(im_arr, [0.33, 0.33, 0.33])

    """ convert the output from the prediction to notes and create a midi file
        from the notes """
    offset = 0
    output_notes = []
    

    prev_notes = updateNotes(im_arr[0,:],{})
    for column in im_arr.T[1:,:]:
        notes = column2notes(column,min_pitch,max_pitch)
        # pattern is a chord
        notes_in_chord = notes
        old_notes = prev_notes.keys()
        for old_note in old_notes:
            if not old_note in notes_in_chord:
                new_note = note.Note(old_note,quarterLength=prev_notes[old_note])
                new_note.storedInstrument = instrument.Piano()
                if offset - prev_notes[old_note] >= 0:
                    new_note.offset = offset - prev_notes[old_note]
                    output_notes.append(new_note)
                elif offset == 0:
                    new_note.offset = offset
                    output_notes.append(new_note)                    
                else:
                    print(offset,prev_notes[old_note],old_note)

        prev_notes = updateNotes(notes_in_chord,prev_notes)

        # increase offset each iteration so that notes do not stack
        offset += resolution

    
    for old_note in prev_notes.keys():
        new_note = note.Note(old_note,quarterLength=prev_notes[old_note])
        new_note.storedInstrument = instrument.Piano()
        new_note.offset = offset - prev_notes[old_note]

        output_notes.append(new_note)

    prev_notes = updateNotes(notes_in_chord,prev_notes)
    
    midi_stream = stream.Stream(output_notes)
    mm1 = tempo.MetronomeMark(number=80)
    midi_stream.insert(0,mm1)

    # Saving music21 stream as midi
    path_list = image_path.split("/")
    midi_path = os.path.join(reduce(os.path.join,path_list[:-1]),path_list[-1].replace(".png","_PP.mid"))
    midi_stream.write('midi', fp=midi_path)

    # Converting MIDI files to .wav
    fs = FluidSynth('../Data/MuseScore_General.sf2')
    fs.midi_to_audio(midi_path, midi_path.replace('.mid','.wav'))
    if play: fs.play_midi(midi_path)

def generate_animation(img_list,output_dir,epochs):

    # fig = plt.figure(figsize=(8,8))
    fig,ax = plt.subplots(figsize=(8,8))
    plt.axis("off")
    # ims = [[plt.imshow(np.transpose(im,(1,2,0)), animated=True),
            # ax.text(0.5,1.05,f'{epochs[i]}', size=plt.rcParams["axes.titlesize"],ha="center", transform=ax.transAxes, )] 
            # for i,im in enumerate(img_list)]
    ims = [plt.imshow(np.transpose(im,(1,2,0)), animated=True) for i,im in enumerate(img_list)]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)
    print(ani)

    writergif = animation.PillowWriter(fps=4) 
    # writervideo = animation.FFMpegWriter(fps=60)
    ani.save(os.path.join(output_dir,'animation.gif'), writer=writergif)


