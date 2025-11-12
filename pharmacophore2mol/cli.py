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
    """
    Pharmacophore2Mol: Command Line Interface (CLI)
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
def evaluate(input_path):
    """
    Evaluate molecules from SDF or XYZ file(s).
    
    Automatically converts XYZ to SDF using OpenBabel if needed.
    
    Examples:
    
        >>> p2m evaluate xyz_dir
        >>> p2m evaluate molecules.sdf

    """
    from .metrics.utils import evaluate_from_file
    
    logger.debug(f"Starting evaluation for: {input_path}")
    
    results = evaluate_from_file(input_path)
    results.print_summary()
    
    logger.debug("Evaluation complete")


@cli.command()
@click.argument('input_path', type=click.Path(exists=True, dir_okay=True, readable=True))
@click.argument('output_path', type=click.Path(writable=True))
@click.option('--keep-disconnected', is_flag=True, help='Keep molecules with disconnected fragments')
@click.option('--keep-unstable', is_flag=True, help='Ignore molecular stability checks')
def clean(input_path, output_path, keep_disconnected, keep_unstable):
    """
    Clean molecule dataset by removing invalid molecules.
    
    Removes molecules that fail rdkit parsing or have deeper structural issues (valence, formal charges, disconnected components).
    Consolidates multiple input files into a single output SDF.
    Supports SDF, XYZ, MOL2, PDB formats (single or multi-file).
    
    Examples:
    
        p2m clean molecules.sdf cleaned.sdf
        
        p2m clean some/dir/with/files/ cleaned.sdf

        p2m clean molecules.sdf cleaned.sdf --keep-disconnected
    """
    from pharmacophore2mol.data.preprocessing import clean_molecules

    logger.debug(f"Starting cleaning: {input_path} to {output_path}")

    num_valid = clean_molecules(
        input_path=input_path,
        output_path=output_path,
        remove_disconnected=not keep_disconnected,
        only_stable=not keep_unstable
    )

    logger.info(f"Cleaning complete: {num_valid} valid molecules saved")