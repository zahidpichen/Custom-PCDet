import glob
import os
from pathlib import Path
import zipfile
from model import PCDet

# Define folder paths
models_dir = 'models'
configs_dir = 'config'
frames_dir = 'frames'
zip_dir = 'zip'

# Ensure zip directory exists
os.makedirs(zip_dir, exist_ok=True)

# Get all model files
model_files = sorted(glob.glob(f'{models_dir}/*.pth'))

# Process each model
for model_path in model_files:
    # Extract model name without .pth extension (e.g., "PartA2")
    model_basename = Path(model_path).stem
    cfg_file = f"{configs_dir}/{model_basename}.yaml"
    
    # Check if matching config exists
    if not os.path.exists(cfg_file):
        print(f"Warning: No matching config found for {model_basename}, skipping.")
        continue
    
    print(f"Processing model: {model_basename}")
    
    # Get all bin files
    bin_files = sorted(glob.glob(f'{frames_dir}/*.bin'))
    txt_files = []
    
    # Process each frame with current model
    for bin_path in bin_files:
        detector = PCDet(
            cfg_file=cfg_file,
            ckpt=model_path,
            data_path=bin_path
        )
        results = detector.detect()  # list with one frame_result

        out_txt = Path(bin_path).with_suffix('.txt')
        txt_files.append(str(out_txt))
        
        with open(out_txt, 'w') as f:
            fr = results[0]
            f.write(f"Source file: {bin_path}\n")
            f.write(f"Total objects: {fr['num_objects']}\n\n")
            for obj in fr['objects']:
                f.write(f"- Label: {obj['label_name']}\n")
                f.write(f"  Score: {obj['score']:.4f}\n")
                f.write(f"  Box: {obj['box']}\n\n")
        print(f"Wrote {out_txt}")
    
    # Create zip file for all txt files
    zip_path = f"{zip_dir}/{model_basename}.zip"
    
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for txt_file in txt_files:
            # Add file to zip with just the filename (not the full path)
            zipf.write(txt_file, os.path.basename(txt_file))
    
    print(f"Created zip file: {zip_path}")
