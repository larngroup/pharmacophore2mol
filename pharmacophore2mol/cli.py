import os
import click
import logging

from importlib.metadata import version, PackageNotFoundError
from pharmacophore2mol.logging_config import setup_logging

try:
    __version__ = version("pharmacophore2mol")
except PackageNotFoundError:
    # Package not installed, mark as development version
    __version__ = "dev"

logger = logging.getLogger(__name__)

@click.group()
@click.version_option(version=__version__, prog_name='pharmacophore2mol')
@click.option('-v', '--verbose', is_flag=True, help='Enable verbose output (DEBUG level)')
@click.option('-q', '--quiet', is_flag=True, help='Suppress all output except errors')
@click.pass_context
def cli(ctx, verbose, quiet):
    """Pharmacophore2Mol: Command Line Interface (CLI)
    """
    # Ensure context object exists
    ctx.ensure_object(dict)
    
    # Store verbosity settings in context
    ctx.obj['verbose'] = verbose
    ctx.obj['quiet'] = quiet
    
    # Initialize logging
    setup_logging(verbose=verbose, quiet=quiet)

@cli.command()
@click.argument('input_path', type=click.Path(exists=True, dir_okay=True, readable=True))
@click.pass_context
def evaluate(ctx, input_path):
    """
    Evaluate molecules from SDF or XYZ file(s).
    
    Automatically converts XYZ to SDF using OpenBabel if needed.
    
    Examples:
    
        >>> p2m evaluate xyz_dir
        >>> p2m evaluate molecules.sdf

    """
    from pharmacophore2mol.metrics.utils import evaluate_from_file
    
    logger.debug(f"Starting evaluation for: {input_path}")
    
    # Run evaluation
    results = evaluate_from_file(
        input_path=input_path,
    )

    results.print_summary()
    logger.debug("Evaluation complete")


@cli.command()
@click.argument('input_path', type=click.Path(exists=True, dir_okay=True, readable=True))
@click.argument('output_path', type=click.Path(dir_okay=False, writable=True))
@click.pass_context
def clean_data(ctx, input_path, output_path):
    """
    Clean molecule data by removing invalid and nonstable molecules from SDF, PDB, MOL, MOL2 or even XYZ file(s).
    Compacts it all to a single SDF file, ready to use.
    Automatically converts XYZ to SDF using OpenBabel if needed.
    
    Examples:

        >>> p2m clean-data sdf_dir cleaned.sdf

    """
    from pharmacophore2mol.data.utils import convert_xyz_to_sdf
    from rdkit import Chem
    # Convert if needed
    sdf_path = input_path
    if os.path.isdir(input_path) or input_path.lower().endswith('.xyz'):
        logger.info("Detected XYZ file(s). Converting to SDF using OpenBabel...")
        sdf_path = convert_xyz_to_sdf(input_path)
        logger.info(f"Conversion complete. SDF saved at: {sdf_path}. Proceeding...")

    # # Load molecules
    # suppl = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=False, strictParsing=False)
    
    # # Filter valid molecules
    # valid_mols = [mol for mol in suppl if mol is not None]

    # # Determine output path
    # if output_path is None:
    #     output_path = os.path.splitext(sdf_path)[0] + "_cleaned.sdf"

    # # Save cleaned SDF
    # writer = Chem.SDWriter(output_path)
    # for mol in valid_mols:
    #     writer.write(mol)
    # writer.close()

    # logger.info(f"Cleaned data saved to {output_path} with {len(valid_mols)} valid molecules.")