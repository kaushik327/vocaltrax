#!/bin/bash
echo "Running no regularization baseline..."
python3 synthesize.py general.temporal_regularization=false

echo "Running lambda=0.0001..."
python3 synthesize.py general.temporal_regularization=true general.temporal_lambda=0.0001

echo "Running lambda=0.001..."
python3 synthesize.py general.temporal_regularization=true general.temporal_lambda=0.001

echo "Running lambda=0.01..."
python3 synthesize.py general.temporal_regularization=true general.temporal_lambda=0.01

echo "Running lambda=0.1..."
python3 synthesize.py general.temporal_regularization=true general.temporal_lambda=0.1

echo "Running lambda=1.0..."
python3 synthesize.py general.temporal_regularization=true general.temporal_lambda=1.0

echo "All done!"