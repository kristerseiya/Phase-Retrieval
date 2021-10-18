# Phase-Retrieval
This is an experimental implementaion of phase-retrieval algorithm based on Gerchberg-Saxton and Plug-and-Play ADMM. 

## What is "Phase Retrieval"?
Phase retrieval is a method where we tried to retrieve x (image) from observation |Ax|. A is a measurement matrix. Here, we only look at DFT matrix as our measurement matrix.

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