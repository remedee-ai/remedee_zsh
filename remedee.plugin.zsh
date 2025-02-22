#!/bin/zsh

# This ZSH plugin reads the text from the current buffer
# and uses a Python script to complete the text.
api="openai"

_ZSH_CODEX_REPO=$(dirname $0)

create_completion() {
    # Get the text typed until now.
    local text=$BUFFER
    #local ZSH_CODEX_PYTHON="${ZSH_CODEX_PYTHON:-python3}"
    local ZSH_CODEX_PYTHON="${_ZSH_CODEX_REPO}/.venv/bin/python"

    # if text is empty, then use the fixed completion below
    if [ -z "$text" ]; then
        text="Look at the last command output above and analyze the issue."
        local completion="$ZSH_CODEX_PYTHON $_ZSH_CODEX_REPO/answer_query.py -p\\
\\
\"$text\""
        BUFFER=$completion
        # move the cursor inside the quotes to make it easier to edit
        CURSOR=$((${#completion} - 2))
    else
        local completion=$(echo -n "$text" | $ZSH_CODEX_PYTHON $_ZSH_CODEX_REPO/create_completion.py $CURSOR)
        local text_before_cursor=${BUFFER:0:$CURSOR}
        local text_after_cursor=${BUFFER:$CURSOR}

        # Add completion to the current buffer.
        BUFFER="${text_before_cursor}${completion}${text_after_cursor}"
        
        CURSOR=$((CURSOR + ${#completion}))
    fi
}

# Bind the create_completion function to a key.
zle -N create_completion
# You may want to add a key binding here, e.g.:
# bindkey '^X^E' create_completion
