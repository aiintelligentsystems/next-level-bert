def infer_batch_size_per_device(
    num_devices: int, effective_batch_size: int, batch_size_per_device: int
):
    needed_batch_size_per_device = int(effective_batch_size / num_devices)
    assert needed_batch_size_per_device == effective_batch_size / num_devices

    needed_gradient_accum_steps = 1
    while (needed_gradient_accum_steps * batch_size_per_device < needed_batch_size_per_device) or (
        effective_batch_size / (needed_gradient_accum_steps * num_devices)
        != int(effective_batch_size / (needed_gradient_accum_steps * num_devices))
    ):
        needed_gradient_accum_steps += 1

    resulting_batch_size_per_device = effective_batch_size / (
        needed_gradient_accum_steps * num_devices
    )
    assert resulting_batch_size_per_device <= batch_size_per_device
    assert resulting_batch_size_per_device == int(resulting_batch_size_per_device)

    batch_size_per_device = int(resulting_batch_size_per_device)
    effective_batch_size_per_step = num_devices * batch_size_per_device

    return batch_size_per_device, needed_gradient_accum_steps, effective_batch_size_per_step
