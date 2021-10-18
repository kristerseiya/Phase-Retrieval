# Phase-Retrieval
This is an experimental implementaion of phase-retrieval algorithm based on Gerchberg-Saxton and Plug-and-Play ADMM. 

# What is "Phase Retrieval"?
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
