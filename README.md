# Agri-SOC Offline Leaf Assistant

## Quick start on Windows

1. Extract the entire ZIP file.
2. Make sure Python 3.11 is installed. During installation, enable **Add Python to PATH**.
3. Double-click `RUN_AGRI_SOC.bat`.
4. On the first run, the launcher creates an isolated `.venv` environment and installs
   the compatible Python packages. This may take several minutes and requires internet.
5. When installation finishes, the desktop application opens. On later runs, double-click
   the same BAT file; it checks dependencies and starts the application.

Do not run the `.py` file before the first setup has completed. The launcher avoids
mixing the project packages with globally installed packages, which helps prevent the
NumPy/Pandas/OpenCV binary incompatibility seen in some existing Python environments.

## Main project contents

- `field_assistant.py` — main desktop GUI.
- `train_plantdoc_mobilenet.py` — model architecture and preprocessing imported by GUI.
- `leaf_severity.py` — image/severity utility functions imported by GUI.
- `train_leaf_severity.py` — optional severity model architecture.
- `requirements.txt` — compatible dependency set; NumPy, Pandas and OpenCV are pinned.
- `RUN_AGRI_SOC.bat` — Windows one-click setup and launch script.
- `preprocessed/leaf_aligned42/field_model_runs/combined_mobile42_s42/best.pt`
  — disease-classification checkpoint at the default path expected by the application.
- `preparation_alignment/` — supplied dataset preparation and training scripts.
- `models/experimental_severity_best.pt` — supplied severity checkpoint for reference.
- `original_archives/` — original uploaded ZIP archives.

## Manual setup (optional)

Use Python 3.11:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python field_assistant.py
```

## About the severity checkpoint

The included severity checkpoint has an empty stored `classes` list. It is retained for
reference but is not enabled in the default launch. Do not claim severity prediction is
verified end-to-end based on this checkpoint alone.

## Dataset and retraining

The original training dataset is not required for inference when the trained disease
checkpoint is included.

The supplied preparation/training scripts are included, but the full retraining pipeline
is not self-contained. The preparation scripts refer to helper modules `leaf_canonical.py`
and `leaf_preproc.py` that were not among the uploaded files. The severity evaluation
script also imports additional helper modules not supplied here. Reproducing training
from raw data requires those helpers and the required datasets/manifests/annotations.

## Important scope

This is the supplied **Windows desktop inference application**. It is not an Android
application and does not include a validated local-language advisory/LLM module.
The launcher installs dependencies and starts the GUI; it does not validate agronomic
accuracy. Test with representative leaf photos before presenting model performance.
