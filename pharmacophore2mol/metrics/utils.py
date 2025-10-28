from pharmacophore2mol.metrics.evaluator import EvaluationResults, Evaluator
import logging
import os

logger = logging.getLogger(__name__)

# def compute_all_statistics(data_list, atom_encoder, charges_dic):
#     """Computes all dataset statistics from molecule data."""
#     num_nodes = node_counts(data_list)
#     atom_types = atom_type_counts(data_list, num_classes=len(atom_encoder))
#     bond_types = edge_counts(data_list)
#     charge_types = charge_counts(data_list, num_classes=len(atom_encoder), charges_dic=charges_dic)
#     valency = valency_count(data_list, atom_encoder)
#     bond_lengths = bond_lengths_counts(data_list)
#     angles = bond_angles(data_list, atom_encoder)
    
#     return Statistics(
#         num_nodes=num_nodes, 
#         atom_types=atom_types, 
#         bond_types=bond_types, 
#         charge_types=charge_types,
#         valencies=valency, 
#         bond_lengths=bond_lengths, 
#         bond_angles=angles
#     )


def evaluate_from_file(input_path):
    """Evaluates molecules from a file (SDF or XYZ)."""
    # from pharmacophore2mol.metrics.evaluator import Evaluator
    # from pharmacophore2mol.data.molecule_dataset import MoleculeDataset

    # load data
    # check if input_path is already a sdf or still a xyz or xyz dir that needs converting
    sdf_path = input_path
    if os.path.isdir(input_path) or input_path.lower().endswith('.xyz'):
        logger.info("Detected XYZ file(s). Converting to SDF using OpenBabel...")
        from pharmacophore2mol.data.utils import convert_xyz_to_sdf
        sdf_path = convert_xyz_to_sdf(input_path)
        logger.info(f"Conversion complete. SDF saved at: {sdf_path}")



    evaluator = Evaluator()

    results = evaluator.evaluate(sdf_path)

    return results