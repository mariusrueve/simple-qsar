#!/usr/bin/env python
"""Simple QSAR model using RDKit and SVM."""

import logging
import os

import click
import joblib
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Suppress RDKit warnings
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


def read_smiles(file_path):
    """Reads SMILES strings from a file and returns a list of RDKit molecule objects."""
    molecules = []
    with open(file_path, "r") as file:
        for line in file:
            smile = line.strip()
            mol = Chem.MolFromSmiles(smile)
            if mol is not None:
                molecules.append((smile, mol))
            else:
                logging.warning(f"Invalid SMILES '{smile}' skipped.")
    return molecules


def calculate_descriptors(molecules):
    """Calculates molecular descriptors for a list of RDKit molecule objects."""
    descriptors = []
    for smile, mol in molecules:
        desc = []
        for name, function in Descriptors.descList:
            try:
                value = function(mol)
            except Exception as e:
                logging.debug(f"Descriptor {name} failed for molecule {smile}: {e}")
                value = np.nan
            desc.append(value)
        descriptors.append(desc)
    return np.array(descriptors)


@click.group()
@click.option("--verbose", "-v", count=True, help="Increase verbosity level (-v, -vv, -vvv).")
@click.pass_context
def cli(ctx, verbose):
    """QSAR Model Command Line Interface."""
    # Set logging level based on verbosity
    levels = [logging.WARNING, logging.INFO, logging.DEBUG]
    level = levels[min(len(levels) - 1, verbose)]  # Cap the level at DEBUG
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")
    ctx.obj = {}
    ctx.obj["verbose"] = verbose


@cli.command()
@click.option(
    "--active",
    required=True,
    type=click.Path(exists=True),
    help="Path to file containing SMILES strings of active molecules.",
)
@click.option(
    "--inactive",
    required=True,
    type=click.Path(exists=True),
    help="Path to file containing SMILES strings of inactive molecules.",
)
@click.option(
    "--model-path",
    default="qsar_model.pkl",
    show_default=True,
    help="Path to save the trained QSAR model.",
)
@click.pass_context
def train(ctx, active, inactive, model_path):
    """Train a QSAR model using active and inactive molecules."""
    logging.info("Reading active molecules...")
    active_mols = read_smiles(active)
    logging.info(f"Number of active molecules: {len(active_mols)}")

    logging.info("Reading inactive molecules...")
    inactive_mols = read_smiles(inactive)
    logging.info(f"Number of inactive molecules: {len(inactive_mols)}")

    logging.info("Calculating descriptors...")
    active_desc = calculate_descriptors(active_mols)
    inactive_desc = calculate_descriptors(inactive_mols)

    # Combine data
    X = np.vstack((active_desc, inactive_desc))
    y = np.array([1] * len(active_desc) + [0] * len(inactive_desc))

    # Handle NaN values by replacing them with the mean of the column
    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(X)

    # Create a pipeline with scaling and SVM classifier
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", probability=True, class_weight="balanced")),
        ]
    )

    logging.info("Training the QSAR model...")
    pipeline.fit(X, y)

    # Save the model
    joblib.dump(pipeline, model_path)
    logging.info(f"Model saved to {model_path}")


@cli.command()
@click.option(
    "--input",
    required=True,
    type=click.Path(exists=True),
    help="Path to file containing SMILES strings of molecules to screen.",
)
@click.option(
    "--model-path",
    default="qsar_model.pkl",
    show_default=True,
    help="Path to the trained QSAR model.",
)
@click.option(
    "--output",
    default="predictions.csv",
    show_default=True,
    help="Path to save the prediction results.",
)
@click.option(
    "--threshold",
    default=0.5,
    show_default=True,
    help="Probability threshold for classifying molecules as active.",
)
@click.pass_context
def screen(ctx, input, model_path, output, threshold):
    """Screen new molecules using the trained QSAR model."""
    if not os.path.exists(model_path):
        logging.error(f"Model file '{model_path}' not found. Please train the model first.")
        return

    logging.info("Loading the QSAR model...")
    pipeline = joblib.load(model_path)

    logging.info("Reading molecules to screen...")
    molecules = read_smiles(input)
    smiles_list = [smile for smile, mol in molecules]
    logging.info(f"Number of molecules to screen: {len(molecules)}")

    logging.info("Calculating descriptors...")
    descriptors = calculate_descriptors(molecules)

    # Handle NaN values by replacing them with the mean of the column
    imputer = SimpleImputer(strategy="mean")
    descriptors = imputer.fit_transform(descriptors)

    logging.info("Predicting activity...")
    probabilities = pipeline.predict_proba(descriptors)[:, 1]

    # Classify molecules based on the threshold
    predicted_classes = ["Active" if prob >= threshold else "Inactive" for prob in probabilities]

    # Save predictions
    with open(output, "w") as f:
        f.write("SMILES,Activity_Probability,Predicted_Class\n")
        for smile, prob, cls in zip(smiles_list, probabilities, predicted_classes):
            f.write(f"{smile},{prob},{cls}\n")

    logging.info(f"Predictions saved to {output}")


if __name__ == "__main__":
    cli()
