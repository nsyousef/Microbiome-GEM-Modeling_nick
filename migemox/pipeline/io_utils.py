"""
I/O Utilities for MiGEMox Pipeline

This module provides functions for loading input data (e.g., abundance files),
preparing model data for saving to disk, and general pipeline helpers such as
memory usage tracking. It centralizes file operations and utility functions
to improve code organization and reusability.
"""

import pandas as pd
import psutil
import os
import re
import cobra
from cobra.io import load_matlab_model
from cobra.util import create_stoichiometric_matrix
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from scipy.sparse import csr_matrix
import pickle

def save_cobra_model_pickle(model: cobra.Model, filename: str):
    """
    Save a COBRApy model to a file using pickle.

    Args:
        model (cobra.Model): The COBRA model object to save.
        filename (str): Path to the output pickle file.
    """
    print(f"Saving model to {filename} (this may take some time for large models)")
    with open(filename, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Model saved as pickle to {filename}")

def load_cobra_model_pickle(filename: str) -> cobra.Model:
    """
    Load a COBRApy model from a pickle file.

    Args:
        filename (str): Path to the input pickle file.

    Returns:
        cobra.Model: The loaded COBRA model.
    """
    print(f"Loading model from {filename}")
    with open(filename, "rb") as f:
        model = pickle.load(f)
    print(f"Model loaded from pickle file {filename}")
    return model

def ensure_parent_dir(file_path: str):
    """
    Ensure a parent directory to a file path exists.

    This function creates the directory if it does not exist.

    Args:
        file_path: the path to the file you want to ensure exists
    """
    parent_dir = os.path.dirname(file_path)
    if parent_dir:  # This will be empty string if file is in current directory
        os.makedirs(parent_dir, exist_ok=True)

def print_memory_usage(stage=""):
    """
    Prints the current memory usage (in MB) of the running Python process.

    Args:
        stage (str): Description of the pipeline stage.
    """
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 ** 2)
    print(f"[MEMORY] {stage}: {mem_mb:.2f} MB")

def get_individual_size_name(abun_file_path: str, mod_path: str) -> tuple:
    """
    Reads abundance data from a CSV file and extracts sample names, organisms,
    and extracellular metabolites from associated model files or a dict of already
    loaded models.

    Args:
        abun_file_path: Path to the abundance CSV file.
        mod_path: Path to the directory containing organism model files (.mat).

    Returns:
        A tuple containing:
            - clean_samp_names: List of cleaned sample names (valid Python identifiers).
            - organisms: List of organism names from the abundance file.
            - ex_mets: List of sorted unique extracellular metabolites found in the models.

    Raises:
        ValueError: If there's an error reading the abundance file or loading a model.
        FileNotFoundError: If a model file is not found.
    """
    # Step 1: Read abundance CSV
    df = pd.read_csv(abun_file_path, index_col=0)
    samp_names, organisms = list(df.columns), list(df.index)

    # Step 2: Clean sample names to be valid Python identifiers
    clean_samp_names = []
    for name in samp_names:
        if not name.isidentifier():
            name = re.sub(r'\W', '_', name)
            if not name[0].isalpha():
                name = 'sample_' + name
        clean_samp_names.append(name)

    # Step 3: Load models and extract [e] metabolites
    ex_mets = set()
    for organism in organisms:
        model_file = os.path.join(mod_path, organism + '.mat')
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"Model file not found: {model_file}")
        try:
            model = load_matlab_model(model_file)
        except Exception as e:
            raise ValueError(f"Error loading model {organism}: {e}")

        # Extract extracellular metabolites (assuming suffix '[e]')
        mets = [met.id for met in model.metabolites if met.id.endswith('[e]')]
        ex_mets.update(mets)

    return clean_samp_names, organisms, list(sorted(ex_mets))

def make_community_gem_dict(model, C=None, d=None, dsense=None, ctrs=None):
    """
    Creates a dictionary representation of a cobra model, including additional
    parameters (C, d, dsense, ctrs) for saving in a .mat file format compatible
    with mgPipe/Microbiome Modeling Toolbox.

    Args:
        model (cobra.Model): The cobra model object.
        C (scipy.sparse.csr_matrix, optional): Coupling matrix. Defaults to None.
        d (numpy.ndarray, optional): Right-hand side of coupling constraints. Defaults to None.
        dsense (numpy.ndarray, optional): Sense of coupling constraints ('E', 'L', 'G'). Defaults to None.
        ctrs (list, optional): Names of coupling constraints. Defaults to None.

    Returns:
        dict: A dictionary containing model components and additional parameters
              suitable for `scipy.io.savemat`.
    """
    num_rxns = len(model.reactions)
    num_mets = len(model.metabolites)

    # Objective vector
    c = np.zeros((num_rxns, 1))
    obj_rxn_id = str(model.objective.expression).split('*')[1].split('-')[0].strip()
    for i, rxn in enumerate(model.reactions):
        if rxn.id == obj_rxn_id:
            c[i, 0] = 1.0
            break

    # S matrix
    S = create_stoichiometric_matrix(model)

    # Bounds
    lb = np.array([rxn.lower_bound for rxn in model.reactions], dtype=np.float64).reshape(-1, 1)
    ub = np.array([rxn.upper_bound for rxn in model.reactions], dtype=np.float64).reshape(-1, 1)

    # Met and rxn info
    rxns = np.array([[rxn.id] for rxn in model.reactions], dtype=object)
    rxnNames = np.array([[rxn.name] for rxn in model.reactions], dtype=object)
    metNames = np.array([[met.name] for met in model.metabolites], dtype=object)
    mets = np.array([[met.id] for met in model.metabolites], dtype=object)

    # Sense
    csense = np.array(['E'] * num_mets, dtype='U1')

    # Default coupling matrices
    if C is None: C = csr_matrix((0, num_rxns))
    if d is None: d = np.zeros((0, 1))
    if dsense is None: dsense = np.array([], dtype='<U1')
    if ctrs is None: ctrs = np.array([], dtype=object).reshape(-1, 1)

    C = csr_matrix((0, num_rxns)) if C is None else C
    d = np.zeros((0, 1)) if d is None else d
    dsense = np.array([], dtype='<U1') if dsense is None else dsense
    ctrs = np.array([], dtype=object).reshape(-1, 1) if ctrs is None else ctrs.reshape(-1, 1)

    # Model name
    model_name = np.array([model.name], dtype=object)
    osenseStr = np.array(['max'], dtype='U3')

    return {
        'rxns': rxns,
        'rxnNames': rxnNames,
        'mets': mets,
        'metNames': metNames,
        'S': S,
        'b': np.zeros((num_mets, 1)),
        'c': c,
        'lb': lb,
        'ub': ub,
        'metChEBIID': np.array([
            m.annotation['chebi'][0].replace('CHEBI:', '') if 'chebi' in m.annotation and isinstance(m.annotation['chebi'], list)
            else m.annotation['chebi'].replace('CHEBI:', '') if 'chebi' in m.annotation and isinstance(m.annotation['chebi'], str)
            else ''
            for m in model.metabolites
        ], dtype=object).reshape(-1, 1),
        'metCharges': np.array([m.charge if getattr(m, 'charge', None) is not None else np.nan for m in model.metabolites]).reshape(-1, 1),
        'metFormulas': np.array([m.formula if getattr(m, 'formula', None) is not None else np.nan for m in model.metabolites]).reshape(-1, 1),
        'rules': np.array([getattr(r, 'gene_reaction_rule', '') for r in model.reactions], dtype=object).reshape(-1, 1),
        'subSystems': np.array([getattr(r, 'subsystem', '') for r in model.reactions], dtype=object).reshape(-1, 1),
        'osenseStr': osenseStr,
        'csense': csense,
        'C': C,
        'd': d,
        'dsense': dsense,
        'ctrs': ctrs,
        'name': model_name
    }

def collect_flux_profiles(samp_names: List[str], exchanges: List[str],
                          net_production: Dict[str, Dict[str, float]],
                          net_uptake: Dict[str, Dict[str, float]],
                          res_path: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Converts net production and net uptake dictionaries, which contain results
    for each sample, into structured Pandas DataFrames.

    Args:
        res_path: path to results directory where flux.csv files should be stored
        samp_names: List of sample identifiers.
        exchanges: List of exchange reaction IDs (metabolites) analyzed.
        net_production: Dictionary where keys are sample names and values are
                        dictionaries of metabolite-to-flux for net production.
        net_uptake: Dictionary where keys are sample names and values are
                    dictionaries of metabolite-to-flux for net uptake.

    Returns:
        Tuple of:
            - net_secretion_df (pd.DataFrame): DataFrame of net production (secretion)
                                               fluxes with metabolites as index and samples as columns.
            - net_uptake_df (pd.DataFrame): DataFrame of net uptake fluxes with
                                            metabolites as index and samples as columns.
    """
    # Ensure exchanges are sorted for consistent DataFrame indexing
    exchanges_sorted = sorted(exchanges)

    res_path = Path.cwd() / 'Results' if not res_path else Path(res_path)
    logger.info(f"Results will be saved to: {res_path}")

    # Prepare data for net production DataFrame
    prod_data, uptk_data = {}, {}
    for samp in samp_names:
        prod_data[samp] = [net_production.get(samp, {}).get(r, 0.0) for r in exchanges_sorted]
        uptk_data[samp] = [net_uptake.get(samp, {}).get(r, 0.0) for r in exchanges_sorted]
    
    net_secretion_df = pd.DataFrame(prod_data, index=exchanges_sorted)
    net_uptake_df = pd.DataFrame(uptk_data, index=exchanges_sorted)

    net_secretion_df.to_csv(res_path / 'inputDiet_net_secretion_fluxes.csv', index=True, index_label='Net secretion')
    net_uptake_df.to_csv(res_path / 'inputDiet_net_uptake_fluxes.csv', index=True, index_label='Net uptake')


    return net_secretion_df, net_uptake_df

def extract_positive_net_prod_constraints(csv_path: str, threshold: float = 0) -> Dict[str, Dict[str, float]]:
    """
    Reads a net secretion/production CSV and returns a dict of
    {met_id: {model_name: flux_value}} for positive fluxes.
    """
    df = pd.read_csv(csv_path, index_col=0)
    # keep only rows with at least one non-zero flux
    mask = (df.iloc[:, 1:].abs() > threshold).any(axis=1)
    filtered = df[mask]
    filtered.index = filtered.index.to_series().apply(lambda x: x.split("[")[0].replace("EX_", ""))
    return filtered.dropna(how="all", axis=1).to_dict(orient="index")
