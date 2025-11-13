"""
generation_TGAN.py
Safe mini-batch synthetic data generation for TimeGAN.
Allows large datasets to be processed without memory or index errors.
"""

def safe_generation(model, num_samples, batch_size, verbose=True):
    """
    Generate synthetic data safely using mini-batches.

    Args:
        model: trained TimeGAN model
        num_samples: total number of sequences to generate
        batch_size: number of samples to generate per batch (default=64)
        verbose: print progress info (default=True)

    Returns:
        List of all generated synthetic sequences
    """
    generated_all = []
    n_batches = (num_samples + batch_size - 1) // batch_size

    if verbose:
        print(f"🧩 Generating {num_samples} samples in {n_batches} mini-batches of {batch_size}...")

    for b in range(n_batches):
        start = b * batch_size
        end = min((b + 1) * batch_size, num_samples)
        current_n = end - start

        # use real sequence lengths for this block
        model.T = model.ori_time[start:end]

        # generate one mini-batch
        gen_batch = model.generation(num_samples=current_n)
        generated_all.extend(gen_batch)

        if verbose:
            print(f"  ✅ Batch {b+1}/{n_batches} generated ({current_n} samples)")

    return generated_all
