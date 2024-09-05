# Image Perturbation and Metric Evaluation

This project evaluates various image quality metrics under different types of perturbations. It applies various noise types to input images and calculates the average metric values between the original and perturbed images.

## Features

- Supports multiple noise types: random perturbation, salt and pepper, blur, Gaussian, black rectangles, and swirl
- Applies perturbations at different intensity levels
- Calculates average metric values for each image, noise type, and perturbation level
- Saves perturbed images for visual inspection
- Exports results to CSV files for further analysis

## Evaluation Scripts

The project includes multiple evaluation scripts for different metrics:

### Perceptual Metrics
- `evaluate_content_error_average.py`: Evaluates Content Error
- `evaluate_ssim_average.py`: Evaluates Structural Similarity Index (SSIM)
- `evaluate_lpips_average.py`: Evaluates Learned Perceptual Image Patch Similarity (LPIPS)

### Style Fidelity Metrics
- `evaluate_style_error_average.py`: Evaluates Style Error
- `evaluate_sifid_average.py`: Evaluates SIFID (Structure Inception Feature Inception Distance)
- `evaluate_artfid_average.py`: Evaluates ArtFID

Each script follows a similar structure but is tailored to its specific metric.

## Requirements

- Python 3.x
- PyTorch
- torchvision
- scikit-image
- PIL
- matplotlib
- OpenCV
- numpy
- scipy
- piq (for SSIM calculation)
- ArtFID (for ArtFID calculation)
- [Add other dependencies as needed]

## Usage

Run the desired evaluation script with the following command:

```
python <script_name>.py --image-dir <path_to_image_directory> [options] --runs <number_of_runs_to_average>
```

### Common Arguments

- `--image-dir`: Directory containing input images (required)
- `--cuda`: Use CUDA if available (default: 1)
- `--image-size`: Size to resize input images (default: 512)
- `--runs`: Number of runs for random noise types (default: 5)
- `--output-dir`: Directory to save perturbed images (optional)

## Output

- `<metric>_results.csv`: CSV file containing metric values for each image, noise type, and perturbation level
- Perturbed images saved in the specified output directory (if `--output-dir` is provided)

## Notes

- The scripts support PNG image files
- For random noise types, the scripts perform multiple runs and calculate the average metric value
- The highest perturbation level for each noise type is visualized and saved (if `--output-dir` is provided)

