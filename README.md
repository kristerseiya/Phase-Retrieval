# Phase Retrieval with Plug-and-Play ADMM
This is an experimental implementaion of phase-retrieval algorithm based on Gerchberg-Saxton and Plug-and-Play ADMM.

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
       --noise 5 \
       --samprate 4 \
       --iter 100 \
       --display \
       --save
```
```markdown
python phre_demo.py -h

  --image IMAGE             path to image
  --hioiter HIOITER         number of iterations of HIO (initialization)
  --pnpiter PNPITER         total number of iterations with PnP-ADMM
  --noise NOISE             the noise level added to the measurement
  --samprate SAMPRATE       oversampling rate for the measurement
  --display DISPLAY         display result if given
  --save SAVE               save result if given
```
