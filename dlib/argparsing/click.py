import os
import re
from typing import Any, List, Union

from click.core import Context, Option

from ..misc.ddict import ddict


class UnlimitedNargsOption(Option):
    """
    When used like `@click.option('--<your option>', cls=UnlimitedNargsOption)`, the option will behave like `nargs=*` of the argparse package.
    From https://stackoverflow.com/a/48394004/10917436
    """

    def __init__(self, *args, **kwargs):
        self.save_other_options = kwargs.pop("save_other_options", True)
        nargs = kwargs.pop("nargs", -1)
        assert nargs == -1, "nargs, if set, must be -1 not {}".format(nargs)
        super(UnlimitedNargsOption, self).__init__(*args, **kwargs)
        self._previous_parser_process = None
        self._eat_all_parser = None

    def add_to_parser(self, parser, ctx):
        def parser_process(value, state):
            # method to hook to the parser.process
            done = False
            value = [value]
            if self.save_other_options:
                # grab everything up to the next option
                while state.rargs and not done:
                    for prefix in self._eat_all_parser.prefixes:
                        if state.rargs[0].startswith(prefix):
                            done = True
                    if not done:
                        value.append(state.rargs.pop(0))
            else:
                # grab everything remaining
                value += state.rargs
                state.rargs[:] = []
            value = tuple(value)

            # call the actual process
            self._previous_parser_process(value, state)

        retval = super(UnlimitedNargsOption, self).add_to_parser(parser, ctx)
        for name in self.opts:
            our_parser = parser._long_opt.get(name) or parser._short_opt.get(name)
            if our_parser:
                self._eat_all_parser = our_parser
                self._previous_parser_process = our_parser.process
                our_parser.process = parser_process
                break
        return retval


def process_click_args(ctx: Context, cmd_args: dict[str, Any]) -> ddict:
    """
    Utility function to use when passing click command line args into a PyTorch Lightning Trainer and when using Weights & Biases.
    Performs validation checks for CUDA availability if GPUs are specified.
    """
    kdict_cmd_args = ddict(cmd_args)
    if kdict_cmd_args.offline:
        os.environ["WANDB_MODE"] = "dryrun"

    ######### Handle CUDA & GPU stuff #########
    if kdict_cmd_args.gpus is not None:
        import torch

        if not torch.cuda.is_available():
            ctx.fail("GPUs were requested but machine has no CUDA!")
        # torch.cudnn.enabled = True
        # torch.cudnn.benchmark = True
        kdict_cmd_args.strategy = "ddp"
        print(
            f"Training on GPUs: {kdict_cmd_args.gpus if kdict_cmd_args.gpus != -1 else 'All available'}"
        )
        # experienced "deadlock" bug with the standard nccl backend
        # os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
    else:
        print("Training on CPU")
        kdict_cmd_args.strategy = None
    return kdict_cmd_args


def int_sequence(cmd_input: List[str]) -> Union[List[int], int]:
    """Accepts the following when used in combination with `cls=UnlimitedNargsOption`:
    - Space-separated numbers like `0 1 2 4 7` -> `[0, 1, 2, 4, 7]`
    - Comma-separated numbers like `0,1,2,4,7` -> `[0, 1, 2, 4, 7]`
    - Range of numbers like `0-2` -> `[0, 1, 2]`
    - Combination of any of the above `0-2,4 6,8-10` -> `[0, 1, 2, 4, 6, 8, 9, 10]`
    """
    if len(cmd_input) == 1 and cmd_input[0] == "-1":
        return -1
    parsed_ints = []
    range_regex = re.compile(r"^(\d+)-(\d+)$")
    split_by_comma = [
        idx_or_range for chunk in cmd_input for idx_or_range in chunk.split(",")
    ]
    for idx_or_range in split_by_comma:
        range_match = range_regex.match(idx_or_range)
        if range_match:
            parsed_ints.extend(
                list(range(int(range_match.group(1)), int(range_match.group(2)) + 1))
            )
        elif idx_or_range.isnumeric():
            parsed_ints.append(int(idx_or_range))
        else:
            raise Exception(
                f"Error while parsing int sequence for cmd line option: \nraw: {cmd_input} \nsplit: {split_by_comma}"
            )
    return sorted(parsed_ints)
