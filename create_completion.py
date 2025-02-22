#!/usr/bin/env python3

import argparse
import sys
import logging
import re

# init logging to file in current directory
logging.basicConfig(filename='remedee.log', level=logging.DEBUG)

from remedee.service_providers.services import ClientFactory

def escape_exclamation_in_quotes(completion):
    # Regular expression to find text within double quotes
    def replace_exclamations(match):
        # Replace '!' with '\!' only inside the matched quoted text
        return match.group(0).replace('!', r'\!')

    # Use a regex to match quoted sections and apply the replacement function
    result = re.sub(r'\"(.*?)\"', replace_exclamations, completion)
    return result

def main():
    # completion = '#!/bin/bash\necho "1!"'

    # # parse the completion and replace all "!" with "\!"
    # completion = escape_exclamation_in_quotes(completion)

    # sys.stdout.write(completion)
    # return

    parser = argparse.ArgumentParser(
        description="Generate command completions using AI."
    )
    parser.add_argument(
        "cursor_position", type=int, help="Cursor position in the input buffer"
    )
    args = parser.parse_args()
    logging.debug(f"cursor_position: {args.cursor_position}")

    client = ClientFactory.create()

    # Read the input prompt from stdin.
    buffer = sys.stdin.read()
    
    zsh_prefix = "#!/bin/zsh\n\n"
    buffer_prefix = buffer[: args.cursor_position]
    buffer_suffix = buffer[args.cursor_position :]
    full_command = zsh_prefix + buffer_prefix + buffer_suffix

    completion = client.get_completion(full_command)

    if completion.startswith(zsh_prefix):
        completion = completion[len(zsh_prefix) :]

    line_prefix = buffer_prefix.rsplit("\n", 1)[-1]
    # Handle all the different ways the command can be returned
    for prefix in [buffer_prefix, line_prefix]:
        if completion.startswith(prefix):
            completion = completion[len(prefix) :]
            break

    if buffer_suffix and completion.endswith(buffer_suffix):
        completion = completion[: -len(buffer_suffix)]

    completion = completion.strip("\n")
    if line_prefix.strip().startswith("#"):
        completion = "\n" + completion

    # parse the completion and replace all "!" within quotes with "\!"
    completion = escape_exclamation_in_quotes(completion)

    sys.stdout.write(completion)


if __name__ == "__main__":
    main()
