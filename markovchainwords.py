#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from sys import stderr, stdout
import string
import argparse
from collections import defaultdict
import numpy as np
from scipy.stats import poisson


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

## What scales of information to encode?

# 1. letter succession
# 2. syllable succession (markov chain of order 3?)
# 3. etymological semantic units (needs dictionary for learning)
# 3. taxonomic rank (genus/species)
# 4. taxonomic abundance: if there are more rodent species, it would be nice to have per-clade transition matrices (new names should look like rodent names more frequently).
# 
## What algorithmic solutions?
#
# - HHMM (Hierarchical HMM): how to define the higher level states?


def learn_letters(infile, maxwords=2, maxwordlen=100):
    s = 97 # ascii start: letter 'a' will be 0.
    a = 26 # length of the alphabet: ascii end - ascii start

    word_ends = set()
    transition_counts = [np.zeros((a+1, a+1))] * maxwords
    word_lengths = np.zeros((maxwords, maxwordlen))

    with open(infile) as f:
        for line in f:
            line = line.lower()
            word_i = -1
            prev_start = 0 # for word length.
            prev_char = ' '
            for char_i, char in enumerate(line):
                i = ord(prev_char)  - s
                j = ord(char) - s
                prev_char = char
                # prev_char not a word character
                if not (0 <= i < a):
                    i = a
                    # if char also not, then go to next pair of characters.
                    if not (0 <= j <a):
                        continue
                    # Otherwise, it means it's a new word, increment.
                    word_i += 1
                    # But skip words beyond the maximum.
                    if word_i>=maxwords:
                        break

                if not (0 <= j < a):
                    j = a
                    word_ends.add(char)
                    
                    word_len = min(char_i - prev_start, maxwordlen)
                    word_lengths[word_i, word_len] += 1

                #print_if_verbose(word_i, char_i, repr(line[char_i]))
                transition_counts[word_i][i, j] += 1

    unknown_ends = word_ends - set(string.whitespace + string.punctuation)
    if unknown_ends:
        print("Warning: unknown characters used as word boundary: " + \
                ' '.join(repr(ch) for ch in unknown_ends), file=stderr)

    P_word_len = word_lengths.cumsum(axis=1) / word_lengths.sum(axis=1, keepdims=True)
    state_counts = [tc.sum(axis=1, keepdims=True) for tc in transition_counts]
    print_if_verbose(state_counts)
    eq_freqs = [sc / sc.sum() for sc in state_counts]
    #Transition rate matrix
    Qs = [tc / sc for tc,sc in zip(transition_counts, state_counts)]
    return Qs, P_word_len

def generate_word_seq(Qs, a=26, s=97, rep=1, P_stop=None):
    # TODO: respect expected word lengths:
    # - Draw a random word length from the empirical distribution.
    # - Draw next letter conditionned on its possible ending of not.
    assert all((np.inf not in Q) for Q in Qs)
    if P_stop is None:
        P_stop = np.stack([poisson.sf(np.arange(70, 10))] * len(Qs))
    
    max_iter = 100
    states = list(range(a+1))
    letters = [chr(s+x) for x in states]
    letters[-1] = ' '

    output = [''] * len(Qs)

    for word_i, Q in enumerate(Qs):
        state = a #* np.ones(rep)
        i = 0
        while i < max_iter:
            # Proba of next letter, with proba of word end dependent on position:
            p = Q[state, :]
            p[:a] *= (1 - P_stop[word_i, i])/p[:a].sum()
            p[a] = P_stop[word_i, i]

            state = np.random.choice(states, p=p)
            if state < a:
                print_if_verbose(letters[state], end='')
                #stdout.flush()
                output[word_i] += letters[state]
            else:
                print_if_verbose(' ', end='')
                break
            i += 1
    return ' '.join(output)


def deterministic_formatting(strdata):
    return strdata.capitalize()


def main(infile, rep=10, maxwords=2):
    Qs, P_stop = learn_letters(infile, maxwords)
    print_if_verbose(np.array2string(Qs[0], precision=2, max_line_width=180))
    for r in range(rep):
        print(deterministic_formatting(generate_word_seq(Qs, rep=rep, P_stop=P_stop)))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('infile')
    parser.add_argument('-r', '--rep', type=int, default=10)
    parser.add_argument('-w', '--maxwords', type=int, default=2)
    parser.add_argument('-v', '--verbose', action='store_true')
    
    dictargs = vars(parser.parse_args())
    VERBOSE = dictargs.pop('verbose')
    print_if_verbose = def_verbosity()

    main(**dictargs)

