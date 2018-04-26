#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from sys import stderr, stdout
import string
import argparse
from collections import defaultdict
import numpy as np


VERBOSE=True

def muted(*args, **kwargs):
    pass

def def_verbosity():
    if VERBOSE:
        return print
    else:
        return muted

print_if_verbose = def_verbosity()

# Learning

## Define how to recognize syllables:

# - [starting sequence of consonants]
# - sequence of vowels/diphtongs
# - [ending sequence of consonants]
# allow hyphen?

## Only make a letter-transition matrix

# Indexed using the ascii number - 97 (the minimum value: 'a')
# + word boundary state 27
# + letters with accents

## Fill syllable transition matrix (sparse matrix?)

# Create vector of learned syllables, and index the matrix with it.

# Generate random Genus + species

def learn_letters(infile, maxwords=2):
    s = 97 # ascii start: letter 'a' will be 0.
    a = 26 # length of the alphabet: ascii end - ascii start

    word_ends = set()
    transition_counts = [np.zeros((a+1, a+1))] * maxwords
    word_lengths = defaultdict(int)

    with open(infile) as f:
        for line in f:
            line = line.lower()
            word_i = 0
            prev_start = 0 # for word length.

            for char_i in range(len(line) - 1):
                i = ord(line[char_i])  - s
                j = ord(line[char_i + 1]) - s
                if not (0 <= i < a):
                    i = a
                    if not (0 <= j <a):
                        continue
                    word_i += 1
                    if word_i>=maxwords:
                        break

                if not (0 <= j < a):
                    j = a
                    word_ends.add(line[char_i + 1])
                    word_lengths[char_i - prev_start] += 1

                print_if_verbose(word_i, char_i, repr(line[char_i]))
                transition_counts[word_i][i, j] += 1


    unknown_ends = word_ends - set(string.whitespace + string.punctuation)
    if unknown_ends:
        print("Warning: unknown characters used as word boundary: " + \
                ' '.join(repr(ch) for ch in unknown_ends), file=stderr)
    state_counts = [tc.sum(axis=1, keepdims=True) for tc in transition_counts]
    eq_freqs = [sc / sc.sum() for sc in state_counts]
    #Transition rate matrix
    Qs = [tc / sc for tc,sc in zip(transition_counts, state_counts)]
    return Qs

def generate_word_seq(Qs, a=26, s=97, rep=1):
    # TODO: respect expected word lengths:
    # - Draw a random word length from the empirical distribution.
    # - Draw next letter conditionned on its possible ending of not.
    assert all((np.inf not in Q) for Q in Qs)

    max_iter = 100
    states = list(range(a+1))
    letters = [chr(s+x) for x in states]
    letters[-1] = ' '

    output = [''] * len(Qs)

    for word_i, Q in enumerate(Qs):
        state = a #* np.ones(rep)
        i = 0
        while i < max_iter:
            state = np.random.choice(states, p=Q[state, :])
            if state < a:
                print_if_verbose(letters[state], end='')
                #stdout.flush()
                output[word_i] += letters[state]
            else:
                print_if_verbose(' ', end='')
                break
            i += 1
    return ' '.join(output)

def main(infile, rep=10):
    Qs = learn_letters(infile)
    print_if_verbose(np.array2string(Qs[0], precision=2, max_line_width=180))
    for r in range(rep):
        print(generate_word_seq(Qs, rep=rep))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('infile')
    parser.add_argument('-r', '--rep', type=int, default=10)
    parser.add_argument('-v', '--verbose', action='store_true')
    
    dictargs = vars(parser.parse_args())
    VERBOSE = dictargs.pop('verbose')
    print_if_verbose = def_verbosity()

    main(**dictargs)

