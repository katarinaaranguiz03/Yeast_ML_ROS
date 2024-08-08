#!/bin/bash

# Activate the conda environment
source /opt/bifxapps/miniconda3/etc/profile.d/conda.sh
unset $PYTHONPATH

conda activate ml_pipeline

# Input filename passed as the first argument to the script, put my path here
input_filename=$1

#base_filename=$(basename $input_filename)

# The new filename that will be created by ML_preprocess.py
# Line 200: save_name = args.df.replace('.txt','') + '_mod.txt'
modified_filename="${input_filename%.txt}_mod.txt"


# clean data
python ML_preprocess.py -df $input_filename -na_method drop


# define test set
python test_set.py -df $modified_filename -type c -p 0.1 -save test_genes.txt

# feature selection
python Feature_Selection.py -df $modified_filename \
                            -test test_genes.txt \
                            -type c  \
                            -alg rf  \
                            -n 50 \
                            -save top_feat_RF.txt

python Feature_Selection.py -df $modified_filename \
                            -test test_genes.txt  \
                            -type c  \
                            -alg random  \
                            -n 10  \
                            -save rand_feat.txt
# classify
python ML_classification.py -df $modified_filename \
                            -test test_genes.txt \
                            -alg rf \
                            -n 100 \
                            -plots T \
                            -feat top_feat_RF.txt_50 \
                            -save run_output

conda deactivate
