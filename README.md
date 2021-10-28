# Phase Retrieval with Plug-and-Play ADMM
This is an experimental implementaion of phase-retrieval algorithm based on Plug-and-Play ADMM.

## What is "Phase Retrieval"?
Phase retrieval is a method where we tried to retrieve x (image) from observation |Ax|. A is a measurement matrix. Here, we only look at DFT matrix as our measurement matrix.

![Alt text](result/ghost_azrael_phre.png?raw=true "GT vs. HIO vs. PnP")
Phase retrieval of Fourier magnitude with added Gaussian noise σ = 5.
(Left) Ground truth (Middle) HIO 2000 iterations (Right) proposed algorithm.

## Command Line

```bash
python phre_demo.py \
       --image imgs/tiger_224x224.png \
       --hioiter 50 \
       --pnpiter 600 \
       --noise gaussian \
       --noiselvl 5 \
       --samprate 4 \
       --iter 100 \
       --display \
       --save tiger_recon.png
```
```markdown
python phre_demo.py -h

  --image IMAGE             path to image
  --hioiter HIOITER         number of iterations of HIO (initialization)
  --pnpiter PNPITER         total number of iterations with PnP-ADMM
  --noise NOISE             noise type ('gaussian' or 'poisson')
  --noiselvl NOISELVL       standard deviation if noise type is gaussian,
                            if poisson, the product of this parameter and the signal
                            would be used as a standard deviation of gaussian to
                            simulate a poisson noise
  --samprate SAMPRATE       oversampling rate for the measurement
  --display DISPLAY         display result if given
  --save SAVE               save result if given a filename
```
