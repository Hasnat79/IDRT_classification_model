#!/bin/bash
#sbatch --get-user-env=L                #replicate login env

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=blip2_classifier_class6      #Set the job name to "JobExample4"
#SBATCH --time=18:30:00              #Set the wall clock limit to 1hr and 30min
#SBATCH --nodes=1                #Request 1 node
#SBATCH --ntasks-per-node=8        #Request 8 tasks/cores per node
#SBATCH --mem=20G                     #Request 16GB per node 
#SBATCH --output=blip2_classifier_class6_Out.%j      #Send stdout/err to "Example4Out.[jobID]"
#SBATCH --gres=gpu:rtx:1          #Request 2 GPU per node can be 1 or 2
#SBATCH --partition=gpu              #Request the GPU partition/queue


cd /scratch/user/hasnat.md.abdullah/IDRT_classification_model/src
python blip2_classifier.py
