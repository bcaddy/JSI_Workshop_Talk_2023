# Abstract

Title: Very High Spatial Resolution MHD Galaxy Simulations with Cholla

The impact of magnetic fields on galactic outflows and winds is not well understood. Modeling these effects has proven challenging in large part due to the cost of MHD simulations: across the field, different compromises between simulation cost, resolution,  and robustness lead to substantially different results. This tension is particularly noticeable in winds where the effects of magnetic fields in simulations range from non-existent to dramatic. To resolve this tension and accurately model the galactic dynamo and magnetic fields in the circumgalactic medium (CGM) we need high resolution and robust MHD simulations. Simulations like this have historically been prohibitively computationally expensive. In this talk, I will describe my work to add magnetohydrodynamics to Cholla, a massively parallel, GPU accelerated, code for modeling astrophysical fluid dynamics. With Cholla MHD, and the new Frontier supercomputer, we can robustly simulate a Milky Way like galaxy with resolution elements of a few parsecs in size throughout the entire galaxy and CGM and resolve this tension.

# Notes

- [ ] “Caddy & Schneider in prep” everywhere

# Outline

- [x] Set expectations, this is a software talk.
- [x] What exactly is Cholla? Focus on it being a tool that is FOSS and works for lots of problems, we just happen to be using it for galactic outflows
- [x] What are GPUs, how do they work, and why do you care?
- [x] Scientific Motivation, galaxy evolution on parsec scales. Evan is talking the most about this so I can just reference it
  - [x] Sims will generate a lot of data that will be public
- [ ] MHD
  - [ ] Talk about MHD. What makes our method good? CT, linear wave convergence plots. PLMC, PPMC
    - [x] why is method good?
    - [ ] PPMC vs. PLMC, LWC + Brio & Wu
  - [ ] Performance. Scaling plots. Highlight that adding more physics doesn’t have a huge impact on performance
  - [ ] Exascale OTV. Include zoom in from full size to individual pixels
- [x] Discuss testing and CI. makes it easier for others to modify and sets apart
- [x] Self plug, looking for an RSE or similar job. Do they need a computationally heavy post-doc or something
