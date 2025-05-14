#!/bin/bash
cd ../
mkdir 'data/Stanford ORB' && cd 'data/Stanford ORB'
wget https://downloads.cs.stanford.edu/viscam/StanfordORB/blender_HDR.tar.gz
wget https://downloads.cs.stanford.edu/viscam/StanfordORB/ground_truth.tar.gz
tar -xvf blender_HDR.tar.gz
tar -xvf ground_truth.tar.gz
rm blender_HDR.tar.gz
rm ground_truth.tar.gz