import numpy as np
# import matplotlib.pyplot as plt
import corsikaio as co
from copy import deepcopy
import subprocess
from collections.abc import Iterable

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


def iter_input_card(variables, runnum0):
    """
    Generate different input card of CORSIKA
    according to given template input file or str.
    Return new cards with combinations of variables.

    Parameters
    ----------
    template, str
    The base template to be modified.

    variables, dict
    Dictionary containing key-value pairs.
    Value can be list type.

    Return
    ------
    inter_cards, iter
    Iterator containing all possible input cards.
    """
    run_number = runnum0
    cards_list = []
    based_card = InputCard("/home/tian/corsika_wrapper/example.inp")
    for zenith in variables:
        theta_min, theta_max = zenith, zenith
        based_card.pars["RUNNR"] = " " + str(run_number)
        based_card.pars["THETAP"] = f" {zenith} {zenith}"
        cards_list.append(deepcopy(based_card))
        run_number += 1
    inter_cards = iter(cards_list)
    return inter_cards


def corsika_run_wrapper(primE, runnum0):
    cards = iter_input_card(primE, runnum0)
    muon_ratio = np.zeros(len(primE))
    muonm = np.zeros(len(primE), dtype=int)
    muonp = np.zeros(len(primE), dtype=int)
    print("Running corsika...")
    for i, card in enumerate(cards):
        runnum = int(card.pars['RUNNR'])
        f = open(card.pars['DIRECT'].strip() + f"DAT{runnum:06d}.lst", "w")
        subprocess.run("corsika76400Linux_QGSJET_gheisha",
                        input=card.get_card().encode('ascii'),
                        stdout=f,
                        stderr=subprocess.STDOUT)
        f.close()
        # e / mu
        muonp[i] = len(read_particle(card.pars['DIRECT'].strip()\
                                    + f"DAT{runnum:06d}", pid=[2, 3])) # e+, e-
        muonm[i] = len(read_particle(card.pars['DIRECT'].strip()\
                                    + f"DAT{runnum:06d}", pid=[5, 6])) # mu+, mu-
        muon_ratio[i] =  muonp[i] / muonm[i]
        print(f"{primE[i]:.3f}, e/mu={muon_ratio[i]:.3f}, {muonp[i] + muonm[i]} particles total.")
    # np.savetxt(card.pars['DIRECT'].strip() + "e_muon_ratio.csv", 
    #         np.vstack((primE, muonp, muonm, muon_ratio)).T,
    #         delimiter=',')


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
        # Print input card without SEED rows
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
    corsika_run_wrapper(np.arange(0, 41, 5, dtype=int), runnum0=9000)

