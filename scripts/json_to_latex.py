import json
import os
from pathlib import Path
import sys
from tabulate import tabulate

def generate_latex_tables(directory):
    # Iterate over subdirectories
    baseline = None
    for subdir, _, files in os.walk(directory):
        json_files = [f for f in files if f.endswith('.json')]

        if 'baseline.json' in files:
            with open(os.path.join(subdir, 'baseline.json'), 'r') as f:
                baseline = json.load(f)['scores_stats']['light']

        if json_files:
            # Initialize a dictionary to store data from JSON files in the same directory
            combined_data = []

            best = {
            # metric: (row_idx, init_value, direction: 1 = higher is better, -1 = lower is better)
                'psnr_hdr': [0, -1, 1],
                'psnr_ldr': [0, -1, 1],
                'ssim': [0, -1, 1],
                'lpips': [0, -1, -1],
                'time': [0, -999, -1]
            }
            if baseline is not None:
                combined_data.append(['Baseline', f"${baseline['psnr_hdr'][0]:.3f}$", f"${baseline['psnr_ldr'][0]:.3f}$", f"${baseline['ssim'][0]:.3f}$", f"${baseline['lpips'][0]:.3f}$", f"${baseline['time'][0]:.2f}$s"])
                for key in best.keys():
                    if baseline[key][0] * best[key][2] > best[key][1]:
                        best[key][0] = 0
                        best[key][1] = baseline[key][0] * best[key][2]

            for i, file in enumerate(sorted(json_files)):
                if not file == 'baseline.json' or baseline is None:
                    json_file_path = os.path.join(subdir, file)
                    with open(json_file_path, 'r') as f:
                        data = json.load(f)['scores_stats']['light']

                    # Merge data from JSON files into combined_data dictionary
                    combined_data.append([file[:-5].replace('_', ' ').capitalize(), f"${data['psnr_hdr'][0]:.3f}$", f"${data['psnr_ldr'][0]:.3f}$", f"${data['ssim'][0]:.3f}$", f"${data['lpips'][0]:.3f}$", f"${data['time'][0]:.2f}$s"])
                    for key in best.keys():
                        if data[key][0] * best[key][2] > best[key][1]:
                            best[key][0] = i + (baseline is not None)
                            best[key][1] = data[key][0] * best[key][2]

            for i, key in enumerate(best.keys()):
                combined_data[best[key][0]][i + 1] = f"$\\mathbf{{{best[key][1] * best[key][2]:.2f}}}$s" if key == 'time' else f"$\\mathbf{{{best[key][1] * best[key][2]:.3f}}}$"

            # Convert combined_data to table format
            table = tabulate(combined_data, headers=['', 'PSNR-H$\\uparrow$', 'PSNR-L$\\uparrow$', 'SSIM$\\uparrow$', 'LPIPS$\\downarrow$', 'Time'], tablefmt='latex_raw', showindex='never', floatfmt='.3f', stralign='center', numalign='center')

            # Write table to LaTeX file
            latex_file_path = os.path.join(subdir, f'{Path(subdir).stem}.tex')
            with open(latex_file_path, 'w') as f:
                f.write(table)

# Example usage
if __name__ == '__main__':
    config_folder = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    assert config_folder.exists() and config_folder.is_dir()

    generate_latex_tables(config_folder)
