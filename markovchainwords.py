#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from sys import stderr, stdout
import string
import argparse
import os.path
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
# + word boundary state 26
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


# Etymological resource:
# https://en.wikipedia.org/wiki/List_of_commonly_used_taxonomic_affixes
# https://en.wikipedia.org/wiki/List_of_Latin_and_Greek_words_commonly_used_in_systematic_names

#def key_to_state()
#def iter_states(sequence)

def enumerate_chunks(sequence, n=1):
    N = len(sequence)
    i = 0
    while i < N:
        yield i, sequence[i:(i+n)]
        i += n
#       yield prev_state, state


def learn_letters(infile, maxwords=2, maxwordlen=100, order=1):
    s = 97 # ascii start: letter 'a' will be 0.
    a = 26 # length of the alphabet: ascii end - ascii start
    valid_wordchars = string.ascii_letters

    word_ends = set()
    transition_counts = [np.zeros(((a+1)**order, a+1)) for _ in range(maxwords)]
    word_lengths = np.zeros((maxwords, maxwordlen))
    
    state_index_maker = (a+1)**np.arange(order) # represent number in base a.

    with open(infile) as f:
        for line in f:
            line = line.lower()
            word_i = -1
            prev_start = 0 # for word length.
            prev_states = ' ' * order
            for state_i, state in enumerate(line):
                prev_is = np.array([ord(ch) for ch in prev_states]) - s
                j = ord(state) - s

                prev_states = prev_states[1:] + state
                nonword = (prev_is <= 0) | (prev_is > a)
                if nonword.any():
                    prev_is[nonword] = a
                    # erase memory of previous word. It's a new chain now.
                    prev_is[:np.argmin(nonword)] = a

                # prev char not a word character
                if not (0 <= prev_is[-1] < a):
                    # if state also not, then go to next char.
                    if not (0 <= j < a):
                        continue
                    # Otherwise, it means it's a new word, increment.
                    word_i += 1
                    # But skip words beyond the maximum.
                    if word_i>=maxwords:
                        break

                if not (0 <= j < a):
                    j = a
                    word_ends.add(state)
                    
                    word_len = min(state_i - prev_start, maxwordlen)
                    word_lengths[word_i, word_len] += 1

                #print_if_verbose(word_i, char_i, repr(line[char_i]))
                i = (state_index_maker * prev_is).sum()
                if state_i == 0:
                    assert i == (a+1)**order - 1, "i: %d; %d; prev_is: %s" % (i, (a+1)**order, prev_is)
                try:
                    transition_counts[word_i][i, j] += 1
                except IndexError:
                    print('prev_is:', prev_is,
                            'nonword:', nonword,
                            'test prev char:', (not (0<=prev_is[-1]<a)),
                            'word_i:', (type(word_i), word_i),
                            'i:', (type(i), i),
                            'j:', (type(j), j), file=stderr)
                    raise

    unknown_ends = word_ends - set(string.whitespace + string.punctuation)
    if unknown_ends:
        print("Warning: unknown characters used as word boundary: " + \
                ' '.join(repr(ch) for ch in unknown_ends), file=stderr)

    P_word_len = word_lengths.cumsum(axis=1) / word_lengths.sum(axis=1, keepdims=True)
    state_counts = [tc.sum(axis=1, keepdims=True) for tc in transition_counts]
    # Study observed/non-observed states
    states = ['']
    for o in range(order):
        states = [s + chr(97+x) if x<a else s + ' ' \
                    for x in range(a+1) for s in states]
    states = np.array(states)
    print(states.shape)
    for sc in state_counts:
        print(sc.shape)
        print('#Observed: %d; #Notseen: %d' % ((sc>0).sum(),
                                                (sc==0).sum()))
        print(states[sc == 0])

    print_if_verbose(state_counts)
    eq_freqs = [sc / sc.sum() for sc in state_counts]
    #Transition rate matrix
    Qs = [tc / sc for tc,sc in zip(transition_counts, state_counts)]
    return Qs, P_word_len

def generate_word_seq(Qs, a=26, s=97, rep=1, P_stop=None, order=1):
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
        prev_states_is = [a] * order
        prev_states_i = ((a+1) ** order - 1) #* np.ones(rep)
        #print(a+1, order, prev_states_i, type(prev_states_i))
        i = 0
        while i < max_iter:
            # Proba of next letter, with proba of word end dependent on position:
            p = Q[prev_states_i, :]
            if np.isnan(p).any():
                assert np.isnan(p).all()
                print(prev_states_is, prev_states_i, i)
                output[word_i] += '.' # THIS SHOULD NOT HAPPEN.
                break
            p[:a] *= (1 - P_stop[word_i, i])/p[:a].sum()
            p[a] = P_stop[word_i, i]

            state = np.random.choice(states, p=p)
            prev_states_is = prev_states_is[1:] + [state]
            prev_states_i = ((a+1) ** np.arange(order) * np.array(prev_states_is)).sum()
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


def main(infile, rep=10, maxwords=2, order=1):
    dbbase = os.path.splitext(os.path.basename(infile))[0]
    end = '-o%d.npy' % order
    dbfile = dbbase + end
    if not os.path.exists(dbfile):
        Qs, P_stop = learn_letters(infile, maxwords, order=order)
        print_if_verbose(np.array2string(Qs[0], precision=2, max_line_width=180))
        np.save(dbfile, Qs)
        np.save(dbfile.replace(end, '-pstop' + end), P_stop)
    else:
        Qs = np.load(dbfile)
        P_stop = np.load(dbfile.replace(end, '-pstop' + end))

    for r in range(rep):
        print(deterministic_formatting(
                generate_word_seq(Qs, rep=rep, P_stop=P_stop, order=order)))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('infile')
    parser.add_argument('-r', '--rep', type=int, default=10)
    parser.add_argument('-w', '--maxwords', type=int, default=2)
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-o', '--order', type=int, default=1)
    
    dictargs = vars(parser.parse_args())
    VERBOSE = dictargs.pop('verbose')
    print_if_verbose = def_verbosity()

    main(**dictargs)

