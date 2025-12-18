"""
Constraints Management for MiGEMox Pipeline

This module centralizes the formulation, pruning, and application of
metabolic constraints, especially the biomass coupling constraints,
to cobra models. It ensures that complex constraints are consistently
defined and integrated into the optimization problems.
"""

import cobra
from cobra_structural import Model as StructuralModel
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.sparse import csr_matrix, vstack, hstack
from optlang import Constraint, Variable, Objective, Model as OptModel
from tqdm import tqdm
from migemox.pipeline.io_utils import print_memory_usage, log_with_timestamp
from datetime import datetime, timezone

# Default Coupling Factor (Used in https://doi.org/10.4161/gmic.22370)
COUPLING_FACTOR = 400


def build_global_coupling_constraints(model: StructuralModel, microbe_list: list[str], coupling_factor: float=400):
    """
    Build sparse coupling constraint matrix for microbe biomass relationships.
    
    Biological Constraint: v_reaction ≤ coupling_factor × v_biomass
    
    Rationale: Individual reaction rates cannot exceed microbe biomass production
    by more than the coupling factor. Prevents unrealistic flux distributions
    where microbe have high metabolic activity but low biomass.
    
    Args:
        model: Community model with all microbe reactions
        microbe_list: List of microbe identifiers  
        coupling_factor: Biomass coupling strength (default from config)
        
    Returns:
        Coupling matrix (C), bounds (d), constraint sense (dsense), names (ctrs)
    """
    print(f"{datetime.now(tz=timezone.utc)}: Building global coupling constraints...")
    rxn_id_to_index = {r.id: i for i, r in enumerate(model.reactions)}

    all_constraints = []
    all_d = []
    all_dsense = []
    all_ctrs = []

    for microbe in microbe_list:

        print(f"{datetime.now(tz=timezone.utc)}: Microbe: {microbe}")
        print("Memory usage:")
        print_memory_usage()
        # Find microbe reactions and biomass reaction
        microbe_rxns = [r for r in model.reactions if r.id.startswith(microbe + '_')]
        biomass_rxns = [r for r in microbe_rxns if 'biomass' in r.id.lower()]

        if not biomass_rxns:
            continue

        biomass_rxn = biomass_rxns[0]
        biomass_idx = rxn_id_to_index[biomass_rxns[0].id]

        for rxn in microbe_rxns:
            if rxn.id == biomass_rxn.id:
                continue  # Don't couple biomass to itself

            rxn_idx = rxn_id_to_index[rxn.id]

            # Create sparse constraint: v_rxn - 400*v_biomass <= 0
            row_indices = [rxn_idx, biomass_idx]
            row_data = [1.0, -coupling_factor]
            constraint_row = csr_matrix((row_data, ([0, 0], row_indices)), shape=(1, len(model.reactions)))

            all_constraints.append(constraint_row)
            all_d.append(0.0)
            all_dsense.append('L')  # <= constraint
            all_ctrs.append(f"slack_{rxn.id}")

            # Also add reverse constraint: v_rxn + 400*v_biomass >= 0 (for reversible reactions)
            if rxn.lower_bound < 0:
                row_indices_rev = [rxn_idx, biomass_idx]
                row_data_rev = [1.0, coupling_factor]
                constraint_row_rev = csr_matrix((row_data_rev, ([0, 0], row_indices_rev)), shape=(1, len(model.reactions)))

                all_constraints.append(constraint_row_rev)
                all_d.append(0.0)
                all_dsense.append('G')
                all_ctrs.append(f"slack_{rxn.id}_R")

    if all_constraints:
        C = vstack(all_constraints)
        d = np.array(all_d).reshape(-1, 1)
        dsense = np.array(all_dsense, dtype='<U1')
        ctrs = np.array(all_ctrs, dtype=object)
    else:
        C = csr_matrix((0, len(model.reactions)))
        d = np.zeros((0, 1))
        dsense = np.array([], dtype='<U1')
        ctrs = np.array([], dtype=object)
    print(f"{datetime.now(tz=timezone.utc)}: Completed building global coupling constraints.")
    print_memory_usage()

    return C, d, dsense, ctrs

def prune_coupling_constraints_by_microbe_fast(
    global_rxn_ids: list[str],
    global_C: csr_matrix,
    global_d: np.ndarray,
    global_dsense: np.ndarray,
    global_ctrs: np.ndarray,
    present_microbe: list[str],
    sample_model
) -> tuple:
    """
    Prunes the global coupling constraints (C, d, dsense, ctrs) to match the
    microbe present in the sample-specific model.

    Args:
        global_rxn_ids (list[str]): The list of all the reaction IDs in the global model.
        global_C (csr_matrix): The global coupling matrix.
        global_d (np.ndarray): The global 'd' vector.
        global_dsense (np.ndarray): The global 'dsense' vector.
        global_ctrs (list): The global constraint names.
        present_microbe (list[str]): List of microbe present in the sample.
        sample_model (cobra_structural.Model): The sample-specific model after pruning microbe.

    Returns:
        tuple: A tuple containing the pruned:
            - C (csr_matrix): Sample-specific coupling matrix.
            - d (np.ndarray): Sample-specific 'd' vector.
            - dsense (np.ndarray): Sample-specific 'dsense' vector.
            - ctrs (list): Sample-specific constraint names.
    """

    print("Dimensions of matrices loaded in prune_coupling_constraints_by_microbe:")
    print(f"global_C: {global_C.shape}")
    print(f"global_d: {global_d.shape}")
    print(f"global_dsense: {global_dsense.shape}")
    print(f"global_ctrs: {global_ctrs.shape}")

    present_microbe_set = set(present_microbe)
    slack_prefix = "slack_"
    keep_rows = []

    # keep_rows: find constraints belonging to microbes that are present
    for i, ctr_name in enumerate(global_ctrs):
        # Extract microbe name from constraint name (e.g., "slack_Bacteroides_sp_2_1_33B_IEX_12ppd_S[u]tr")
        if ctr_name.startswith(slack_prefix):
            microbe_part = ctr_name[len(slack_prefix):]
            for microbe in present_microbe_set:
                if microbe_part.startswith(microbe):
                    keep_rows.append(i)
                    break

    log_with_timestamp(f"Length of keep_rows: {len(keep_rows)}")

    if keep_rows:
        keep_rows = np.asarray(keep_rows, dtype=int)

        # Remap columns to match sample-specific model
        sample_rxn_ids = [r.id for r in sample_model.reactions]
        log_with_timestamp(f"length of sample_rxn_ids: {len(sample_rxn_ids)}")

        global_rxn_idx_map = {rid: i for i, rid in enumerate(global_rxn_ids)}
        log_with_timestamp(f"length of global_rxn_idx_map: {len(global_rxn_idx_map)}")

        # Build ordered list of column indices matching sample_rxn_ids
        # This will raise KeyError if a sample reaction is missing from global_rxn_ids,
        # which is usually what you want (it signals a mismatch).
        col_indices = [global_rxn_idx_map[rid] for rid in sample_rxn_ids]

        # 1. Slice rows
        subC = global_C[keep_rows, :]

        # 2. Slice columns in one shot, in the correct order
        pruned_C = subC[:, col_indices]

        # Vectors can just be sliced by keep_rows
        pruned_d = global_d[keep_rows, :]
        pruned_dsense = global_dsense[keep_rows]
        pruned_ctrs = global_ctrs[keep_rows]

    else:
        pruned_C = csr_matrix((0, len(sample_model.reactions)))
        pruned_d = np.zeros((0, 1))
        pruned_dsense = np.array([], dtype='<U1')
        pruned_ctrs = np.array([], dtype=object)

    return pruned_C, pruned_d, pruned_dsense, pruned_ctrs

def prune_coupling_constraints_by_microbe(
    global_rxn_ids: list[str],
    global_C: csr_matrix,
    global_d: np.ndarray,
    global_dsense: np.ndarray,
    global_ctrs: list,
    present_microbe: list[str],
    sample_model: StructuralModel
) -> tuple:
    """
    Prunes the global coupling constraints (C, d, dsense, ctrs) to match the
    microbe present in the sample-specific model.

    Args:
        global_rxn_ids (list[str]): The list of all the reaction IDs in the global model.
        global_C (csr_matrix): The global coupling matrix.
        global_d (np.ndarray): The global 'd' vector.
        global_dsense (np.ndarray): The global 'dsense' vector.
        global_ctrs (list): The global constraint names.
        present_microbe (list[str]): List of microbe present in the sample.
        sample_model (cobra_structural.Model): The sample-specific model after pruning microbe.

    Returns:
        tuple: A tuple containing the pruned:
            - C (csr_matrix): Sample-specific coupling matrix.
            - d (np.ndarray): Sample-specific 'd' vector.
            - dsense (np.ndarray): Sample-specific 'dsense' vector.
            - ctrs (list): Sample-specific constraint names.
    """

    # print matrices for debugging
    print(f"Dimensions of matrices loaded in prune_coupling_constraints_by_microbe:")
    print(f"global_C: {global_C.shape}")
    print(f"global_d: {global_d.shape}")
    print(f"global_dsense: {global_dsense.shape}")
    print(f"global_ctrs: {global_ctrs.shape}")

    present_microbe_set = set(present_microbe)
    slack_prefix = "slack_"
    keep_rows = []

    for i, ctr_name in enumerate(tqdm(global_ctrs)):
        # Extract microbe name from constraint name (e.g., "slack_Bacteroides_sp_2_1_33B_IEX_12ppd_S[u]tr")
        if ctr_name.startswith(slack_prefix):
            microbe_part = ctr_name[len(slack_prefix):]
            for microbe in present_microbe_set:
                if microbe_part.startswith(microbe):
                    keep_rows.append(i)
                    break
    
    log_with_timestamp(f"Length of keep_rows: {len(keep_rows)}")

    if keep_rows:
          # Remap columns to match sample-specific model
        sample_rxn_ids = [r.id for r in sample_model.reactions]

        log_with_timestamp(f"length of sample_rxn_ids: {len(sample_rxn_ids)}")

        global_rxn_idx_map = {rid: i for i, rid in enumerate(global_rxn_ids)}

        log_with_timestamp(f"length of global_rxn_idx_map: {len(global_rxn_idx_map)}")

        cols = []
        # for rid in sample_rxn_ids:
        for i in range(len(sample_rxn_ids)):
            print(f"Processing reaction {i} of {len(sample_rxn_ids)}")
            rid = sample_rxn_ids[i]
            idx = global_rxn_idx_map.get(rid)
            if idx is not None:
                cols.append(global_C[keep_rows, idx])
            else:
                cols.append(csr_matrix((len(keep_rows), 1)))
        pruned_C = hstack(cols)
        
        pruned_d = global_d[keep_rows, :]
        pruned_dsense = global_dsense[keep_rows]
        pruned_ctrs = global_ctrs[keep_rows]
    else:
        pruned_C = csr_matrix((0, len(sample_model.reactions)))
        pruned_d = np.zeros((0, 1))
        pruned_dsense = np.array([], dtype='<U1')
        pruned_ctrs = np.array([], dtype=object)
    
    return pruned_C, pruned_d, pruned_dsense, pruned_ctrs

def apply_couple_constraints(model: cobra.Model, model_data: dict) -> cobra.Model:
    """
    Applies biomass coupling constraints to a cobra model using optlang.

    This function reads the pre-calculated coupling matrix (C), right-hand side (d),
    and sense vector (dsense) from a .mat file, and adds them as linear
    constraints to the provided cobra model's solver interface.

    Args:
        model (cobra.Model): The cobra model to which constraints will be added.
        model_data: Dictionary containing the coupling data (C, d, dsense, ctrs)

    Returns:
        cobra.Model: The model with added coupling constraints.
    """

    # S = csr_matrix(modelscipy['S'])     # stoichiometric matrix
    # C = csr_matrix(modelscipy.get('C', np.empty((0, S.shape[1]))))  # fallback to empty
    C = csr_matrix(model_data['C'])  # NOTE: used to fallback to matrix of zeros of dimension of stoich matrix if empty, but now does not
    # d = modelscipy['d'].reshape(-1, 1).astype(float)
    d = model_data['d'].reshape(-1, 1).astype(float)

    C_csr = C.tocsr()
    # dsense = modelscipy.get('dsense')
    dsense = model_data['dsense']
    
    forward_vars = np.array([rxn.forward_variable for rxn in model.reactions])
    reverse_vars = np.array([rxn.reverse_variable for rxn in model.reactions])
    constraints_to_add = []

    for i in tqdm(range(C.shape[0])):
        row = C_csr.getrow(i)
        if row.nnz == 0:  # Skip empty rows
            continue
        
        expr = sum(row.data[k] * (forward_vars[row.indices[k]] - reverse_vars[row.indices[k]]) for k in range(row.nnz))
        if dsense[i] == 'E':
            constraints_to_add.append(Constraint(expr, lb=d[i,0], ub=d[i,0]))
        elif dsense[i] == 'L':
            constraints_to_add.append(Constraint(expr, ub=d[i,0]))
        elif dsense[i] == 'G':
            constraints_to_add.append(Constraint(expr, lb=d[i,0]))

    model.add_cons_vars(constraints_to_add)

    return model

# Running FVA by building an optlang model from scratch
# Good for deeply understanding how coupling constraints work
def build_constraint_matrix(model_path):
    """
    Build the constraint matrix A and RHS vector from a scipy model dictionary of a .mat file.
    
    Args:
        model_path (str): Path to the .mat file containing the model.

    Returns:
        A (scipy.sparse.csr_matrix), rhs (numpy.ndarray), csense (numpy.ndarray)
    """

    # Load the model from the .mat file using scipy
    data = loadmat(model_path, simplify_cells=True)
    modelscipy = data["model"]

    S = csr_matrix(modelscipy['S'])     # stoichiometric matrix
    C = csr_matrix(modelscipy.get('C', np.empty((0, S.shape[1]))))  # fallback to empty
    d = modelscipy['d'].reshape(-1, 1).astype(float)
    b = modelscipy['b'].reshape(-1, 1).astype(float)
    csense = modelscipy.get('csense')   # character vector like ['E', 'E', ..., 'L', ...]

    # Final constraint matrix A and RHS
    A = vstack([S, C])                           # shape: (m + k, n)
    rhs = np.vstack([b, d])                      # shape: (m + k, 1)

    # Constraint sense vector
    csense = np.concatenate([modelscipy['csense'].flatten(), modelscipy['dsense'].flatten()])

    lb = modelscipy['lb'].flatten()  # Lower bounds, shape (n,)
    ub = modelscipy['ub'].flatten()  # Upper bounds, shape (n,)
    c = modelscipy['c'].flatten()    # Objective coefficients, shape (n,)
    
    return A, rhs, csense, lb, ub, c

def build_optlang_model(A: csr_matrix, rhs: np.ndarray, csense: np.ndarray,
                       lb: np.ndarray, ub: np.ndarray, c: np.ndarray) -> tuple:
    """
    Build optlang optimization model from constraint matrix components.
    
    This creates an optimization model compatible with multiple solvers
    for performing flux variability analysis on large constraint systems.
    
    Performance Optimization: Batch constraint creation instead of individual loops.
    
    Args:
        A: Sparse constraint matrix A
        rhs: Right-hand side vector
        csense: Constraint sense vector ('E', 'L', 'G')
        lb/ub: Variable lower/upper bounds
        c: Objective coefficient vector
        
    Returns:
        Tuple of (optimization_model, variables, objective_expression)
    """
    n_vars = A.shape[1]
    vars = [Variable(f'v_{i}', lb=lb[i], ub=ub[i]) for i in range(n_vars)]
    opt_model = OptModel()
    opt_model.add(vars)
    A_csr = A.tocsr()  # Ensure CSR format for efficient row access
    
    for batch_start in tqdm(range(0, A.shape[0], 1000), desc="Building constraints"):
        batch_end = min(batch_start + 1000, A.shape[0])
        batch_constraints = []
        
        for i in range(batch_start, batch_end):
            row = A_csr.getrow(i)
            if row.nnz == 0:  # Skip empty rows
                continue
                
            expr = sum(row.data[k] * vars[row.indices[k]] for k in range(row.nnz))
            sense = csense[i]
            
            if sense == 'E':
                constr = Constraint(expr, lb=rhs[i, 0], ub=rhs[i, 0])
            elif sense == 'L':
                constr = Constraint(expr, ub=rhs[i, 0])
            elif sense == 'G':
                constr = Constraint(expr, lb=rhs[i, 0])
            
            batch_constraints.append(constr)
        
        opt_model.add(batch_constraints)
    obj_expr = sum(c[j] * vars[j] for j in range(n_vars))
    return opt_model, vars, obj_expr

def run_sequential_fva(opt_model, vars: list, obj_expr, 
                      rxn_ids: list, opt_percentage: float = 99.99) -> tuple:
    """
    Perform sequential flux variability analysis with optimality constraints.
    
    Biological Context: FVA determines the range of possible flux values for each
    reaction while maintaining near-optimal growth. This reveals metabolic
    flexibility and identifies essential vs. dispensable pathways.

    Args:
        opt_model (optlang.Model): The optimization model.
        vars (list): List of optlang variables.
        obj_expr (optlang.Expression): Objective expression.
        rxn_ids (list): List of reaction IDs to analyze.
        opt_percentage (float): Percentage of optimal objective to constrain FVA.

    Returns:
        min_fluxes (dict), max_fluxes (dict): Minimum and maximum fluxes for each reaction.
    """
    # Get optimal objective value
    opt_model.objective = Objective(obj_expr, direction='max')
    print(f'Model Status after optimization: {opt_model.optimize()}')
    optimal_obj = opt_model.objective.value

    min_flux = []
    max_flux = []
    for j in tqdm(rxn_ids, desc="FVA (sequential)"):
        # Minimize
        opt_model.objective = Objective(vars[j], direction='min')
        if opt_percentage < 100:
            opt_constr = Constraint(obj_expr, lb=opt_percentage/100 * optimal_obj)
            opt_model.add(opt_constr)
        opt_model.optimize()
        min_flux.append(vars[j].primal if opt_model.status == 'optimal' else None)
        if opt_percentage < 100:
            opt_model.remove(opt_constr)
        # Maximize
        opt_model.objective = Objective(vars[j], direction='max')
        if opt_percentage < 100:
            opt_constr = Constraint(obj_expr, lb=opt_percentage/100 * optimal_obj)
            opt_model.add(opt_constr)
        opt_model.optimize()
        max_flux.append(vars[j].primal if opt_model.status == 'optimal' else None)
        if opt_percentage < 100:
            opt_model.remove(opt_constr)
    
    # Using a pd Dataframe with rxn_id as index and min and max as columns, return two dicts
    df = pd.DataFrame({
        'rxn_id': [vars[j].name for j in rxn_ids],
        'min_flux': min_flux,
        'max_flux': max_flux
    }).set_index('rxn_id')

    return df['min_flux'].to_dict(), df['max_flux'].to_dict()
