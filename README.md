# QSAR Model CLI Tool

A command-line interface (CLI) tool for training and using a Quantitative Structure-Activity Relationship (QSAR) model for ligand-based virtual screening. The tool allows users to train a model using known active and inactive molecules and then use the trained model to screen new molecules for potential activity against a specific target.

## Usage

The tool provides two main commands:

    train: Train a QSAR model using active and inactive molecules.
    screen: Screen new molecules using the trained QSAR model.

Training the Model

    python qsar_model.py train --active ACTIVE_FILE --inactive INACTIVE_FILE [OPTIONS]

Screening Molecules

    python qsar_model.py screen --input INPUT_FILE [OPTIONS]

## Options

### Global Options

    --verbose, -v: Increase verbosity level (can be used multiple times for more verbosity).

### Train Command Options

    --active: Path to the file containing SMILES strings of active molecules (one per line). (Required)
    --inactive: Path to the file containing SMILES strings of inactive molecules (one per line). (Required)
    --model-path: Path to save the trained QSAR model. (Default: qsar_model.pkl)

### Screen Command Options

    --input: Path to the file containing SMILES strings of molecules to screen (one per line). (Required)
    --model-path: Path to the trained QSAR model. (Default: qsar_model.pkl)
    --output: Path to save the prediction results. (Default: predictions.csv)
    --threshold: Probability threshold for classifying molecules as active. (Default: 0.5)

The predictions will be saved in predictions.csv with the following columns:

    SMILES: The SMILES representation of the molecule.
    Activity_Probability: The predicted probability of the molecule being active.
    Predicted_Class: The classification ('Active' or 'Inactive') based on the specified threshold.
