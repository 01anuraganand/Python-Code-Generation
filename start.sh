#!/bin/bash

# Get the value of ROOT_DIR argument
ROOT_DIR=$1

# Run the command using ROOT_DIR variable
pip install -r "${ROOT_DIR}/requirements.txt"