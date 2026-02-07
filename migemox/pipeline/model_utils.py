"""
Model Utilities

This file contains generic utilities for handling models.
"""

from cobra import Model
from cobra_structural import Model as StructuralModel

def find_biomass_candidates(
        model: Model | StructuralModel,
        microbe_name: str | None = None,
    ):
    """
    Find candidate reactions that could be the biomass reaction.

    By default, or if `microbe_name = None` this function searches the model for reactions whose ID
    starts with 'bio' (case insensitive). This mirrors the behavior of MMT.

    However, if the reactions in a microbe are tagged with the microbe's name, then this will break
    that functionality, as the biomass reaction ID will no longer start with `bio`. To fix this, you will
    need to pass in the `microbe_name` if that is the case. If a `microbe_name` is passed in, it will be removed
    from any reaction in which it is present before checking if that reaction starts with `bio`. Please note that this
    removal is case insensitive.

    Please note that this function will automatically concatenate an `_` to the end of `microbe_name` before removing the name.
    This is because MiGEMox by convention puts an underscore between the microbe name and the reaction id. For example:

    ```
    microbe_name: E_coli
    biomass_rxn_id: biomass123

    tagged_rxn_name: E_coli_biomass_123
                           ^ notice this underscore
    ```

    Therefore, you should not include this underscore in `microbe_name`. For example, set `microbe_name` to `E_coli` in the
    case above. Do not set it to `E_coli_`.

    **Another note:** Previously, MiGEMox searched for biomass reactions as reactions containing the word `'biomass'` in them. However,
    this is insufficient as some AGORA2 models have biomass reactions named `bio1`. It is also insufficient to just search for reactions
    whose IDs contain the string `'bio'` in them because there are reactions such as `'pbiosynthesis'` which are not the biomass reaction
    and would therefore get wrongly flagged. For that reason, we must search specifically for srings that start with `'bio'`. This is
    the behavior MMT uses.
    
    :param model: The model or structural model to search to find biomass reaction candidates.
    :type model: Model | StructuralModel
    :param microbe_name: The name of the microbe being checked, if reactions are tagged with microbe name. E.g. `E_coli`
    :type microbe_name: str | None
    """

    # get all reaction IDs in model
    rxn_ids = [rxn.id.lower() for rxn in model.reactions]

    # if necessary, remove microbe name (case insensitive)
    if microbe_name is not None:
        rxn_ids = [rxn_id.replace(microbe_name.lower() + '_', '') for rxn_id in rxn_ids]

    # find reaction IDs that start with 'bio'
    biomass_candidates = [rxn_id for rxn_id in rxn_ids if rxn_id.startswith("bio")]

    return biomass_candidates
