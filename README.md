# Extract event-related timeseries

<!-- brainlife-app:begin (generated from the App manifest; edit the manifest, not this block) -->

> **Extracts the BOLD volumes that overlap selected event types from a preprocessed task fMRI run, using the events table and an HRF-convolved regressor from nilearn.**

Selects the BOLD volumes that fall within the chosen event types: event onsets and durations from the BIDS events.tsv (trial_type, onset, duration, weight columns) are convolved with nilearn's SPM HRF regressor and frames with a positive regressor are kept. Use it to isolate event-related volumes before further analysis. Note: main reads the config keys bold and bold_events, while the registered input sends data and bold_events, so the App may fail to find the BOLD image.

| | |
|---|---|
| 📥 **Inputs** | `neuro/func/task` |
| 📤 **Outputs** | `neuro/func/task` |
| 🐳 **Container** | `anibalsolon/blapp-extract-event-signal` |
| ⚙️ **Resources** | 16 CPUs |
| 🔗 **Run it** | [doi.org/10.25663/brainlife.app.803](https://doi.org/10.25663/brainlife.app.803) |

#### 📥 Inputs

| Input | Datatype | Required | Description |
|---|---|---|---|
| `data` | `neuro/func/task` (preprocessed) | yes | Preprocessed task BOLD image with its BIDS events.tsv (needs trial_type, onset, duration and weight columns). |

#### 📤 Outputs

| Output | Datatype | Description |
|---|---|---|
| `output` | `neuro/func/task` | BOLD image containing only the selected volumes (bold.nii.gz) and the filtered events table (events.tsv). |

<details>
<summary><b>⚙️ Parameters</b> (1)</summary>

| Parameter | Type | Default | Description |
|---|---|---|---|
| `events` | string |  | Event names separated by commas. |

</details>

<!-- brainlife-app:end -->

