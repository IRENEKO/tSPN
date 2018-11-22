Deep Model Compression and Inference Speedup of Sum-Product Networks on Tensor Trains for Matlab&copy;/Octave&copy;
--------------------------------------------------------------------------------------------------

This package contains Matlab/Octave code for converting a trained SPN to conpact tSPN.


1. Functions
------------

* demo

Demonstrates the usage of the tspn_iden algorithm in converting trained SPNs to tSPNs. 

* [core,nz,data,testdata]=tspn_iden(tensor,weight,sample_train,sample_test,opts)

Converts an SPN to its tSPN.

* nonrepeated.m

Finds non-repeated samples.

* findnonsample.m

Finds non-samples (negative sampling).


2. Reference
------------
"Deep Model Compression and Inference Speedup of Sum-Product Networks on Tensor Trains"

Authors: Ching-Yun Ko, Cong Chen, Yuke Zhang, Kim Batselier, Ngai Wong


