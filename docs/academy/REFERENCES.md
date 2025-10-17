# REFERENCES

Notes
- This list compiles canonical sources cited across the SNN Academy documents. Where venues or DOIs are uncertain, they are omitted intentionally. Users should verify publisher/venue/DOI details as needed for their bibliographies.
- Core implementation cross-references within this repository appear in the topical documents (e.g., TPP theory, readout, validation) and point to specific functions such as [`Python.nll_continuous()`](src/sr_ciden/losses.py:152), [`Python.odeint_augmented()`](src/sr_ciden/solvers.py:455), [`Python.sample_ogata()`](src/sr_ciden/readout.py:152), and [`Python.time_rescaling_test()`](src/sr_ciden/validation.py:169).

Temporal Point Processes and Simulation
- Hawkes, A. G. (1971). Spectra of some self-exciting and mutually exciting point processes. Biometrika, 58(1), 83–90.
- Lewis, P. A. W., & Shedler, G. S. (1979). Simulation of nonhomogeneous Poisson processes by thinning. Naval Research Logistics Quarterly, 26, 403–413.
- Ogata, Y. (1981). On Lewis–Shedler simulation method for point processes. IEEE Transactions on Information Theory, 27(1), 23–31.
- Brown, E. N., Barbieri, R., Ventura, V., Kass, R. E., & Frank, L. M. (2002). The time-rescaling theorem and its applications to neural spike train data analysis. Neural Computation, 14(2), 325–346.

Neural ODEs and Continuous-Time Learning
- Chen, T. Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D. (2018). Neural Ordinary Differential Equations. Advances in Neural Information Processing Systems (NeurIPS).

Plasticity and Learning in Spiking Systems
- Bi, G. Q., & Poo, M. M. (1998). Synaptic modifications in cultured hippocampal neurons: Dependence on spike timing, synaptic strength, and postsynaptic cell type. Journal of Neuroscience, 18(24), 10464–10472.
- Song, S., Miller, K. D., & Abbott, L. F. (2000). Competitive Hebbian learning through spike-timing-dependent synaptic plasticity. Nature Neuroscience, 3(9), 919–926.
- Neftci, E. O., Mostafa, H., & Zenke, F. (2019). Surrogate gradient learning in spiking neural networks. IEEE Signal Processing Magazine, 36(6), 51–63.

Neuron Models and Foundations
- Dayan, P., & Abbott, L. F. (2001). Theoretical Neuroscience. MIT Press.
- Ermentrout, G. B., & Terman, D. H. (2010). Mathematical Foundations of Neuroscience. Springer.
- Izhikevich, E. M. (2003). Simple model of spiking neurons. IEEE Transactions on Neural Networks, 14(6), 1569–1572.

Frameworks and Simulators
- Stimberg, M., Brette, R., & Goodman, D. F. (2019). Brian 2, an intuitive and efficient neural simulator. eLife, 8:e47314.
- Gewaltig, M.-O., & Diesmann, M. (2007). NEST (NEural Simulation Tool). Scholarpedia, 2(4), 1430.
- Bekolay, T., Bergstra, J., Hunsberger, E., et al. (2014). Nengo: A Python tool for building large-scale functional brain models. Frontiers in Neuroinformatics, 7, 48.
- Mozafari, M., Ganjtabesh, M., Nowzari-Dalini, A., & Thorpe, S. J. (2019). SpykeTorch: Efficient simulation of convolutional spiking neural networks with at most one spike per neuron. (arXiv preprint; verify venue/version as needed.)
- Eshraghian, J. K., Ward, M., Neftci, E. O., et al. (2021). snnTorch: Deep spiking neural networks in Python. IEEE Access, 9, 147294–147308.

Additional Notes
- For Loihi/Lava and SpiNNaker deployment details, consult the official hardware and SDK documentation provided by the vendors and associated ecosystem projects (e.g., Nengo Loihi, Lava).
- When preparing manuscripts, ensure that all references are cross-checked against authoritative sources for correctness of author lists, titles, venues, volumes, and page ranges.