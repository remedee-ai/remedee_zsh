<h1 align="center">⌨️ 🦾Remedee for  Zsx</h1>

<p align="center">
    AI in the command line.
</p>

<p align="center">
    <a href="https://github.com/tom-doerr/zsh_codex/stargazers"
        ><img
            src="https://img.shields.io/github/stars/tom-doerr/zsh_codex?colorA=2c2837&colorB=c9cbff&style=for-the-badge&logo=starship style=flat-square"
            alt="Repository's starts"
    /></a>
    <a href="https://github.com/tom-doerr/zsh_codex/issues"
        ><img
            src="https://img.shields.io/github/issues-raw/tom-doerr/zsh_codex?colorA=2c2837&colorB=f2cdcd&style=for-the-badge&logo=starship style=flat-square"
            alt="Issues"
    /></a>
    <a href="https://github.com/tom-doerr/zsh_codex/blob/main/LICENSE"
        ><img
            src="https://img.shields.io/github/license/tom-doerr/zsh_codex?colorA=2c2837&colorB=b5e8e0&style=for-the-badge&logo=starship style=flat-square"
            alt="License"
    /><br />
    <a href="https://github.com/tom-doerr/zsh_codex/commits/main"
		><img
			src="https://img.shields.io/github/last-commit/tom-doerr/zsh_codex/main?colorA=2c2837&colorB=ddb6f2&style=for-the-badge&logo=starship style=flat-square"
			alt="Latest commit"
    /></a>
    <a href="https://github.com/tom-doerr/zsh_codex"
        ><img
            src="https://img.shields.io/github/repo-size/tom-doerr/zsh_codex?colorA=2c2837&colorB=89DCEB&style=for-the-badge&logo=starship style=flat-square"
            alt="GitHub repository size"
    /></a>
</p>

<p align="center">
    <img src='https://github.com/tom-doerr/bins/raw/main/zsh_codex/zc4.gif'>
    <p align="center">
        You just need to write a comment or variable name and the AI will write the corresponding code.
    </p>
</p>

## What is it?

This is a ZSH plugin that enables you to use AI powered code completion in the command line. It now supports both OpenAI's Codex and Google's Generative AI (Gemini). OpenAI Codex is the AI that also powers GitHub Copilot, while Gemini is Google's advanced language model.

## How do I install it?

### Manual Installation

1. Install the OpenAI package or the Google package.

```bash
pip3 install openai
```

or

```bash
pip3 install google-generativeai
```

2. Download the ZSH plugin.

```bash
git clone https://github.com/tom-doerr/zsh_codex.git ~/.oh-my-zsh/custom/plugins/zsh_codex 
```

3. Add the following to your `.zshrc` file.

Using oh-my-zsh:

```bash
    plugins=(zsh_codex)
    bindkey '^X' create_completion
```

Without oh-my-zsh:

```bash
    # in your/custom/path you need to have a "plugins" folder and in there you clone the repository as zsh_codex
    export ZSH_CUSTOM="your/custom/path"
    source "$ZSH_CUSTOM/plugins/zsh_codex/zsh_codex.plugin.zsh"
    bindkey '^X' create_completion
```

4. Create a file called `zsh_codex.ini` in `~/.config`.
   Example:

```ini
; Primary service configuration
; Set 'service' to match one of the defined sections below.
[service]
service = groq_service

; Example configuration for a self-hosted Ollama service.
[my_ollama]
api_type = openai
api_key = dummy_key
model = llama3.1
base_url = http://localhost:11434/v1

; OpenAI service configuration
; Provide the 'api_key' and specify a 'model' if needed.
[openai_service]
api_type = openai
api_key = <openai_apikey>

; Groq service configuration
; Provide the 'api_key'.
[groq_service]
api_type = groq
api_key = <groq_apikey>
model = gemma2-9b-it

; Mistral service configuration
; Provide the 'api_key'.
[mistral_service]
api_type = mistral
api_key = <mistral_apikey>
model = mistral-small-latest
```

In this configuration file, you can define multiple services with their own configurations. The required and optional parameters of the `api_type` are specified in `services/sevices.py`. Choose which service to use in the `[service]` section.

6. Run `zsh`, start typing and complete it using `^X`!
7. If you use virtual environments you can set `ZSH_CODEX_PYTHON` to python executable where `openai` or `google-generativeai` is installed.
   e.g. for `miniconda` you can use:

```bash
export ZSH_CODEX_PYTHON="$HOME/miniconda3/bin/python"
```

### Fig Installation

<a href="https://fig.io/plugins/other/zsh_codex_tom-doerr" target="_blank"><img src="https://fig.io/badges/install-with-fig.svg" /></a>

## Troubleshooting

### Unhandled ZLE widget 'create_completion'

```
zsh-syntax-highlighting: unhandled ZLE widget 'create_completion'
zsh-syntax-highlighting: (This is sometimes caused by doing `bindkey <keys> create_completion` without creating the 'create_completion' widget with `zle -N` or `zle -C`.)
```

Add the line

```
zle -N create_completion
```

before you call `bindkey` but after loading the plugin (`plugins=(zsh_codex)`).

### Already exists and is not an empty directory

```
fatal: destination path '~.oh-my-zsh/custom/plugins'
```

Try to download the ZSH plugin again.

```
git clone https://github.com/tom-doerr/zsh_codex.git ~/.oh-my-zsh/custom/plugins/zsh_codex
```

---

<p align="center">
    <a href="https://www.buymeacoffee.com/TomDoerr" target="_blank"><img src="https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png" alt="Buy Me A Coffee" style="height: 41px !important;width: 174px !important;box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;-webkit-box-shadow: 0px 3px 2px 0px rgba(190, 190, 190, 0.5) !important;" ></a>
</p>

## More usage examples

<p align="center">
    <img src='https://github.com/tom-doerr/bins/raw/main/zsh_codex/update_insert/all.gif'>
    <p align="center">
    </p>
</p>

---

[Fish Version](https://github.com/tom-doerr/codex.fish)

[Traffic Statistics](https://tom-doerr.github.io/github_repo_stats_data/tom-doerr/zsh_codex/latest-report/report.html)
