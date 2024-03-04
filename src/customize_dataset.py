from datasets.iterable_dataset import _batch_arrow_tables, _convert_to_arrow
from datasets.formatting import TensorFormatter, get_formatter
from datasets.features.features import cast_to_python_objects
import sys
from itertools import islice
from datasets.filesystems import _reset_fsspec_lock
from datasets.utils.logging import get_logger
from datasets.iterable_dataset import _examples_to_batch, _apply_feature_types_on_batch, _apply_feature_types_on_example

logger = get_logger(__name__)


def __iter__(self):
    if "torch" in sys.modules:
        import torch.utils.data

        worker_info = torch.utils.data.get_worker_info()
        if isinstance(self, torch.utils.data.IterableDataset) and worker_info is not None:
            # We're a torch.utils.data.IterableDataset in a PyTorch worker process
            yield from self._iter_pytorch()
            return

    ex_iterable = self._prepare_ex_iterable_for_iteration()
    if self._formatting:
        formatter = get_formatter(self._formatting.format_type, features=self.features)
        format_dict = (
            formatter.recursive_tensorize if isinstance(formatter, TensorFormatter) else cast_to_python_objects
        )
    else:
        format_dict = None

    if self._formatting and (ex_iterable.iter_arrow or self._formatting.format_type == "arrow"):
        if ex_iterable.iter_arrow:
            iterator = _batch_arrow_tables(ex_iterable.iter_arrow(), batch_size=self.batch_size)
        else:
            iterator = _convert_to_arrow(ex_iterable, batch_size=self.batch_size)
        for key, pa_table in iterator:
            yield formatter.format_row(pa_table)
        return

    for key, example in ex_iterable:
        if self.features:
            # `IterableDataset` automatically fills missing columns with None.
            # This is done with `_apply_feature_types_on_example`.
            example = _apply_feature_types_on_example(
                example, self.features, token_per_repo_id=self._token_per_repo_id
            )
        yield format_dict(example) if format_dict else example

def _iter_pytorch(self):
        ex_iterable = self._prepare_ex_iterable_for_iteration()
        # fix for fsspec when using multiprocess
        _reset_fsspec_lock()
        # check if there aren't too many workers
        import torch.utils.data

        worker_info = torch.utils.data.get_worker_info()
        if self._is_main_process() and ex_iterable.n_shards < worker_info.num_workers:
            logger.warning(
                f"Too many dataloader workers: {worker_info.num_workers} (max is dataset.n_shards={ex_iterable.n_shards}). "
                f"Stopping {worker_info.num_workers - ex_iterable.n_shards} dataloader workers."
            )
            logger.info(
                f"To parallelize data loading, we give each process some shards (or data sources) to process. "
                f"Therefore it's unnecessary to have a number of workers greater than dataset.n_shards={ex_iterable.n_shards}. "
                f"To enable more parallelism, please split the dataset in more files than {ex_iterable.n_shards}."
            )
        # split workload
        _log_prefix = f"node#{self._distributed.rank} " if self._distributed else ""
        shards_indices = self._ex_iterable.split_shard_indices_by_worker(worker_info.id, worker_info.num_workers)
        if shards_indices:
            logger.debug(
                f"{_log_prefix}dataloader worker#{worker_info.id}, ': Starting to iterate over {len(shards_indices)}/{ex_iterable.n_shards} shards."
            )
            ex_iterable = ex_iterable.shard_data_sources(worker_id=worker_info.id, num_workers=worker_info.num_workers)

            if self._formatting:
                formatter = get_formatter(self._formatting.format_type, features=self.features)
                format_dict = (
                    formatter.recursive_tensorize if isinstance(formatter, TensorFormatter) else cast_to_python_objects
                )
            else:
                format_dict = None

            if self._formatting and (ex_iterable.iter_arrow or self._formatting == "arrow"):
                if ex_iterable.iter_arrow:
                    iterator = _batch_arrow_tables(ex_iterable.iter_arrow(), batch_size=self.batch_size, drop_last_batch=self.drop_last_batch)
                else:
                    iterator = _convert_to_arrow(ex_iterable, batch_size=self.batch_size, drop_last_batch=self.drop_last_batch)
                if self.batch_size > 1:
                    for key, pa_table in iterator:
                        yield formatter.format_batch(pa_table)
                    return
                else:
                    for key, pa_table in iterator:
                        yield formatter.format_row(pa_table)
                    return
            
            else:
                iterator = iter(ex_iterable)
                if self.batch_size > 1:
                    for key, example in iterator:
                        # If batched, first build the batch
                        examples = [example] + [example for key, example in islice(iterator, self.batch_size - 1)]
                        if self.drop_last_batch and len(examples) < self.batch_size:  # ignore last batch
                            return
                        batch = _examples_to_batch(examples)
                        if self.features:
                            # `IterableDataset` automatically fills missing columns with None.
                            # This is done with `_apply_feature_types_on_batch`.
                            batch = _apply_feature_types_on_batch(batch, self.features, token_per_repo_id=self._token_per_repo_id)
                        yield format_dict(batch) if format_dict else batch
                else:
                    for key, example in ex_iterable:
                        if self.features:
                            # `IterableDataset` automatically fills missing columns with None.
                            # This is done with `_apply_feature_types_on_example`.
                            example = _apply_feature_types_on_example(
                                example, self.features, token_per_repo_id=self._token_per_repo_id
                            )
                        yield format_dict(example) if format_dict else example
            logger.debug(
                f"{_log_prefix}dataloader worker#{worker_info.id}, ': Finished iterating over {len(shards_indices)}/{ex_iterable.n_shards} shards."
            )
        else:
            logger.debug(
                f"{_log_prefix}dataloader worker#{worker_info.id}, ': Stopping... Number of dataset shards < num_workers ({ex_iterable.n_shards}<{worker_info.num_workers})."
            )