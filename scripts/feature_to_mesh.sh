#!/bin/bash

# Check if at least one argument is provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <base_directory> [model] [model_type] [voxel_size]"
    exit 1
fi

BASE_DIR="$1"
MODEL="${2:-ResUNetBN2C-16feat-3conv.pth}"  # Use "ResUNetBN2C-16feat-3conv.pth" if not provided
MODEL_TYPE="${3:-3DMatch}"  # Use "3DMatch" if not provided
VOXEL_SIZE="${4:-0.025}"  # Use 0.025 if not provided
FEATURE_TO_MESH_SCRIPT='feature_to_mesh.py'

# Find all .ply files in the base directory
PLY_FILES=("$BASE_DIR"/*.ply)

# Iterate over the .ply files and run feature_to_mesh.py for each
for PLY_FILE in "${PLY_FILES[@]}"; do
    echo "Processing file: $PLY_FILE"
    
    # Run feature_to_mesh.py for the current file
    python "$FEATURE_TO_MESH_SCRIPT" -i "$PLY_FILE" -m "$MODEL" -mt "$MODEL_TYPE" --voxel_size "$VOXEL_SIZE"
    
    # Check the exit status of the last command
    if [ $? -ne 0 ]; then
        echo "Error processing file: $PLY_FILE"
        exit 1
    fi
done

# If the loop completes without errors, print a success message
echo "Processing completed successfully."
