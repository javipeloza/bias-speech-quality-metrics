# Performance of Objective Speech Quality Metrics on Languages Beyond Validation Data

Code for Bachelor thesis, [_Performance of Objective Speech Quality Metrics on Languages Beyond Validation Data: A Study of Turkish and Korean_](TODO).

Abstract:

> This study investigates the performance of two objective speech quality metrics, Perceptual Evaluation of Speech Quality (PESQ) and Virtual Speech Quality Objective Listener (ViSQOL), in predicting human-rated speech quality scores, which are essential for telecommunication systems' Quality of Experience (QoE). These metrics have been validated using a limited number of languages due to the insufficiency of labeled data with human-rated scores. This research focuses on the applicability of PESQ and ViSQOL on Turkish and Korean, two languages that were not part of the validation data for calibrating these metrics. The experiment used English as the baseline language for comparison, and the results showed that Turkish samples had higher average ViSQOL scores, with the difference being statistically significant compared to the English samples. Furthermore, Turkish male speakers had the highest correlation between PESQ and ViSQOL scores, and ViSQOL rated speech higher than PESQ, especially under babble noise degradations. Future work should focus on exploring biases across additional metrics, languages, gender, and degradation conditions.

## Results

Below are the key visualizations from the experiment:

#### Average PESQ and ViSQOL Scores by Language
<div style="display: flex; justify-content: center; gap: 0px;">  
  <img src="plots/average_pesq_visqol_by_snr.png" alt="Average PESQ and ViSQOL by SNR" width="45%">
  <img src="plots/score_distributions.png" alt="Score Distributions" width="45%">
</div>

#### Score Density Violin Plot & Density Distributions
<div style="display: flex; justify-content: center; gap: 0px;">    
  <img src="plots/score_density_violin.png" alt="Score Density Violin Plot" width="45%">
  <img src="plots/score_density_distributions.png" alt="Score Density Distributions" width="45%">
</div>

#### Scores by Degradation Type & Correlation by Gender
<div style="display: flex; justify-content: center; gap: 0px;">
  <img src="plots/metrics_by_noise_type.png" alt="Metrics By Noise Type" width="45%">
  <img src="plots/metric_correlation_by_gender.png" alt="Metric Correlation by Gender" width="45%">
</div>

#### Kolmogorov-Smirnov (KS) Test Results

<div style="display: flex; justify-content: space-between; gap: 20px;">
  <div>
    <h4>By Language</h4>
    <table>
      <tr>
        <th>Metric</th>
        <th>Comparison</th>
        <th>KS-statistic</th>
        <th>p-value</th>
      </tr>
      <tr>
        <td><strong>PESQ</strong></td>
        <td>English vs Korean</td>
        <td>0.17</td>
        <td>0.61</td>
      </tr>
      <tr>
        <td></td>
        <td>English vs Turkish</td>
        <td>0.20</td>
        <td>0.44</td>
      </tr>
      <tr>
        <td></td>
        <td>Korean vs Turkish</td>
        <td>0.21</td>
        <td>0.29</td>
      </tr>
      <tr>
        <td><strong>ViSQOL</strong></td>
        <td>English vs Korean</td>
        <td>0.14</td>
        <td>0.79</td>
      </tr>
      <tr>
        <td></td>
        <td>English vs Turkish</td>
        <td>0.33</td>
        <td><strong>0.02</strong></td>
      </tr>
      <tr>
        <td></td>
        <td>Korean vs Turkish</td>
        <td>0.23</td>
        <td>0.18</td>
      </tr>
    </table>
  </div>

  <div>
    <h4>By Degradation Type</h4>
    <table>
      <tr>
        <th>Metric</th>
        <th>Comparison</th>
        <th>KS-statistic</th>
        <th>p-value</th>
      </tr>
      <tr>
        <td><strong>PESQ</strong></td>
        <td>Blue vs Pink Noise</td>
        <td>0.12</td>
        <td>0.93</td>
      </tr>
      <tr>
        <td></td>
        <td>Blue vs Babble Noise</td>
        <td>0.17</td>
        <td>0.61</td>
      </tr>
      <tr>
        <td></td>
        <td>Pink vs Babble Noise</td>
        <td>0.19</td>
        <td>0.44</td>
      </tr>
      <tr>
        <td><strong>ViSQOL</strong></td>
        <td>Blue vs Pink Noise</td>
        <td>0.10</td>
        <td>0.99</td>
      </tr>
      <tr>
        <td></td>
        <td>Blue vs Babble Noise</td>
        <td>0.31</td>
        <td><strong>0.04</strong></td>
      </tr>
      <tr>
        <td></td>
        <td>Pink vs Babble Noise</td>
        <td>0.29</td>
        <td>0.06</td>
      </tr>
    </table>
  </div>
</div>


#### Statistical Metrics for PESQ and ViSQOL Score Correlation
| Metric            | Overall | Non-Turkish Male | Turkish Male   | Diff  |
|------------------|---------|--------|------|-------|
| **MAD**          | 0.71    | 0.73   | 0.62 | -0.11 |
| **RMSD**         | 0.89    | 0.91   | 0.77 | -0.13 |
| **Mean difference** | -0.62   | -0.65  | -0.47 | 0.18  |

These plots provide insights into the relationships between different metrics, score distributions, noise type effects, and how SNR affects PESQ and ViSQOL scores.


## Installation

To install the required packages, follow these steps:

1. Install MATLAB (**R2024a** or later) from [here](https://www.mathworks.com/downloads/).
2. Install **Python 3** from [here](https://www.python.org/downloads/).
3. Install the required **Python dependencies** by running:

   ```bash
   pip install -r requirements.txt
   
4. Install the MATLAB Engine for Python by following the instructions [here](https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html).
5. Install [Audio Toolbox](https://mathworks.com/products/audio.html) in MATLAB, as it is required for audio processing.

## Running the Experiment and Plotting Results 

To run the experiment, execute the following command:

  ```bash
  python main.py
  ```

To plot your results and perform statistical analysis, exectute the following command:

  ```bash
  python results_extractor.py
  ```

Note that to plot the results, it is necessary to have an `analysis_results.json` file in the same format as the one included in this repository, which is not generated automatically by the `main.py` file. To generate this file, run `fix_txt_files_to_json` in `file_manager.py` inside each of the individual language results txt file. Then, extract the `results` component and save it to `analysis_results.json`.

## Citation (BibTeX)

Please cite this repository if it was useful for your research:

```bibtex
@article{javipeloza2025speechqualitybias,
  title={Performance of Objective Speech Quality Metrics on Languages Beyond Validation Data: A Study of Turkish and Korean},
  author={Pérez López, Javier},
  year={2025},
  school={Delft University of Technology},
  type={Bachelor Thesis},
  url = {},
}
```
