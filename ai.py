from pathlib import Path
import fastai
from fastai.learner import load_learner
#this comment
# Fix PosixPath issue
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Load the learner
learn = load_learner('export.pkl')

from fastai.vision.all import *
learn=load_learner('export.pkl')
name='catdog'
is_cat,_,probs = learn.predict(PILImage.create(name+'.png'))
print(f" {name} is {is_cat}.")
print(f"Probability that {name} is a cat: {probs[0]:.4f}")
print(f"Probability that {name} is a dog: {probs[1]:.4f}")