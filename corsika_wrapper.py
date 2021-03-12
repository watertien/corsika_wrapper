import numpy as np
# import matplotlib.pyplot as plt
import corsikaio as co
from copy import deepcopy
import subprocess
from collections.abc import Iterable
from multiprocessing import Pool

__author__ = "Tian"


def read_particle(fname, pid=[5,6]):
    """
    Read Coriska particle file and return particle
    of the same type by particel id,
    Format is the same as particle data sub-block.

    Parameters
    ----------
    fname, str
    coriska particle data filename
    
    pid, float or list
    particles to be extracted

    Return
    ------
    out, np.array
    data of one (or several) type of particles
    """
    out = np.array([], dtype=[('particle_description', '<f4'),
                                ('px', '<f4'),
                                ('py', '<f4'),
                                ('pz', '<f4'),
                                ('x', '<f4'),
                                ('y', '<f4'),
                                ('t', '<f4')])

    if not isinstance(pid, Iterable):
        # If input is an integer, convert it to a list
        pid = [pid]

    with co.CorsikaParticleFile(fname) as f:
        for event in f:
            particles = event.particles
            # Get particle type
            ids = particles["particle_description"] // 1000
            out_index = [_id in pid for _id in ids]
            out = np.append(out, particles[out_index])
            # Include mu+ and mu-
            # muons = particles[(ids == 5) | (ids == 6)]
    
    return out


def plot_spectra(energy, bins=30):
    """
    Plot spectra/histogram of given data
    in log/log scale.

    Parameters
    ----------
    energy, np.array
    Energy of secondary particles

    Return
    ------
    fig, ax
    figure and axis objects containing the hist plot
    """
    emin, emax = np.log10([energy.min(), energy.max()])
    log_bins = np.logspace(emin, emax, bins)
    fig, ax = plt.subplots()
    ax.hist(energy, histtype="step", bins=log_bins, log=True)
    ax.set_xscale("log")
    return fig, ax


def iter_input_card(variables, values, runnum0):
    """
    Generate different input card of CORSIKA
    according to given template input file or str.
    Return new cards with combinations of variables.

    Parameters
    ----------
    variables, str
    Name of the parameter to be varied.
    
    values, list
    Values of the parameter.
    
    runnum0, int
    The RUNNR of the first input card.

    Return
    ------
    inter_cards, iterator
    Iterator containing all input cards.
    """
    run_number = runnum0
    cards_list = []
    based_card = InputCard("/home/tian/corsika_wrapper/example.inp")
    for value in values:
        based_card.pars["RUNNR"] = " " + str(run_number)
        based_card.pars[variables] = f" {str(value).strip('[]')}"
        # Use deepcopy to point to different objects in memory
        cards_list.append(deepcopy(based_card))
        run_number += 1
    inter_cards = iter(cards_list)
    return inter_cards


def corsika_run_wrapper(card):
    runnum = int(card.pars["RUNNR"])
    with open(card.pars['DIRECT'].strip() + f"DAT{runnum:06d}.lst", "w") as f:
        subprocess.run("corsika76400Linux_QGSJET_gheisha_SLANT",
                        input=card.get_card().encode('ascii'),
                        stdout=f,
                        stderr=subprocess.STDOUT)


def corsika_run_parallel(name, values, runnum0):
    cards = iter_input_card(name, values, runnum0)
    print("Running corsika...")
    with Pool(processes=len(values)) as p:
        p.map(corsika_run_wrapper, cards)
    print(f"{len(values)} files have been written "
          + f"with RUNNR {runnum0:06d} to {runnum0 + len(values):06d}")
    

class InputCard():
    def __init__(self, fname):
        self.filename = fname
        a = np.genfromtxt(fname, delimiter=',', dtype=str)
        a[:,0] = [i.strip() for i in a[:,0]]
        a[:-1,1] = [np.array(i) for i in a[:-1,1]]
        b = dict(a)
        # Input card without SEED and EXIT kwards
        del b["EXIT"]
        del b["SEED"]
        self.pars = b 

    def display(self):
        # Print input card without SEED and EXIT rows
        for par, value in self.pars.items():
            print(par, value)

    def get_card(self):
        # Add SEED rows and return input card as a string
        np.random.seed(int(self.pars["RUNNR"]))
        seed1, seed2 = np.random.randint(100000, size=2)
        card_str = ""
        for k, v in self.pars.items():
            card_str += k + v + '\n'
        card_str += f"SEED {seed1} 0 0\n"
        card_str += f"SEED {seed2} 0 0\n"
        card_str += f"EXIT"
        return card_str


if __name__ == "__main__":
    par = "THETAP"
    values = np.arange(0, 41, 5).reshape((9, 1)) * np.ones((9, 2))
    corsika_run_parallel(par, values, runnum0=11000)
