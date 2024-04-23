import h5py
import numpy as np
import json
import pcmls
import jax
import argparse


def main():

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run pcmls from matlab script")
    parser.add_argument('input_file', type=str,
                        help='HDF5 file containing input data')
    parser.add_argument('output_file', type=str,
                        help='HDF5 file containing output data')
    parser.add_argument('config_file', type=str,
                        help='JSON file containing configuration params')

    # Parse arguments
    args = parser.parse_args()

    # Configure JAX
    jax.config.update('jax_enable_x64', True)
    # jax.config.update('jax_debug_nans', True)

    # Read settings from JSON
    with open(args.config_file, 'r') as f:
        config = json.load(f)

    # Read input data
    with h5py.File(args.input_file, 'r') as file:
        q = np.array(file['/q'])
        p = np.array(file['/p'])
        fp = np.array(file['/fp'])

    # Extract settings for pcmls from JSON config
    kwargs = config.get("pcmls_kwargs", {})
    kwargs['vectorized'] = bool(kwargs.get('vectorized', False))
    kwargs['compute_gradients'] = bool(kwargs.get('compute_gradients', True))
    kwargs['compute_hessians'] = bool(kwargs.get('compute_hessians', True))
    kwargs['verbose'] = bool(kwargs.get('verbose', False))
    kwargs['m'] = int(kwargs.get('m', 4))

    fq, grad_fq, hess_fq = pcmls.pcmls(q, p, fp, **kwargs)

    # Save results to file
    with h5py.File(args.output_file, 'w') as file:
        file.create_dataset('fq', data=fq)
        file.create_dataset('grad_fq', data=grad_fq)
        file.create_dataset('hess_fq', data=hess_fq)


if __name__ == "__main__":
    main()
