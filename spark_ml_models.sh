#!/bin/bash
#$ -l h_rt=4:00:00  #time needed
#$ -pe smp 20 #number of cores
#$ -l rmem=2G #number of memery
#$ -o Q1_core20_output.txt #This is where your output and errors are logged.
#$ -j y # normal and error outputs into a single file (the file above)
#$ -M zli132@sheffield.ac.uk #Notify you by email, remove this line if you don't like
#$ -m ea #Email you when it finished or aborted
#$ -cwd # Run job from current directory
#$ -P rse-com6012
#$ -q rse-com6012.q

module load apps/java/jdk1.8.0_102/binary

module load apps/python/conda

source activate myspark

spark-submit --driver-memory 8g --executor-memory 2g --master local[10] --conf spark.driver.maxResultSize=4g ../Code/spark_ml_models.py
