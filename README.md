---
license: other
datasets:
- ehartford/wizard_vicuna_70k_unfiltered
language:
- en
tags:
- uncensored
inference: false
---

<!-- header start -->
<div style="width: 100%;">
    <img src="https://i.imgur.com/EBdldam.jpg" alt="TheBlokeAI" style="width: 100%; min-width: 400px; display: block; margin: auto;">
</div>
<div style="display: flex; justify-content: space-between; width: 100%;">
    <div style="display: flex; flex-direction: column; align-items: flex-start;">
        <p><a href="https://discord.gg/Jq4vkcDakD">Chat & support: my new Discord server</a></p>
    </div>
    <div style="display: flex; flex-direction: column; align-items: flex-end;">
        <p><a href="https://www.patreon.com/TheBlokeAI">Want to contribute? TheBloke's Patreon page</a></p>
    </div>
</div>
<!-- header end -->

# Eric Hartford's Wizard-Vicuna-30B-Uncensored GPTQ

This is GPTQ format quantised 4bit models of [Eric Hartford's Wizard-Vicuna 30B](https://huggingface.co/ehartford/Wizard-Vicuna-30B-Uncensored).

It is the result of quantising to 4bit using [GPTQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa).

## Repositories available

* [4bit GPTQ models for GPU inference](https://huggingface.co/TheBloke/Wizard-Vicuna-30B-Uncensored-GPTQ).
* [4bit and 5bit GGML models for CPU inference](https://huggingface.co/TheBloke/Wizard-Vicuna-30B-Uncensored-GGML).
* [float16 HF format model for GPU inference and further conversions](https://huggingface.co/TheBloke/Wizard-Vicuna-30B-Uncensored-fp16).

## How to easily download and use this model in text-generation-webui

Please make sure you're using the latest version of text-generation-webui

1. Click the **Model tab**.
2. Under **Download custom model or LoRA**, enter `TheBloke/Wizard-Vicuna-30B-Uncensored-GPTQ`.
3. Click **Download**.
4. The model will start downloading. Once it's finished it will say "Done"
5. In the top left, click the refresh icon next to **Model**.
6. In the **Model** dropdown, choose the model you just downloaded: `Wizard-Vicuna-30B-Uncensored-GPTQ`
7. The model will automatically load, and is now ready for use!
8. If you want any custom settings, set them and then click **Save settings for this model** followed by **Reload the Model** in the top right.
  * Note that you do not need to set GPTQ parameters any more. These should all be set to default values, as they are now set automatically from the file `quantize_config.json`.
9. Once you're ready, click the **Text Generation tab** and enter a prompt to get started!

## Provided files

**Compatible file - Wizard-Vicuna-30B-Uncensored-GPTQ-4bit.act-order.safetensors**

This will work with all versions of GPTQ-for-LLaMa. It has maximum compatibility

It was created without group_size to minimise VRAM usage, and with `--act-order` to improve inference quality.

* `Wizard-Vicuna-30B-Uncensored-GPTQ-4bit.act-order.safetensors`
  * Works with all versions of GPTQ-for-LLaMa code, both Triton and CUDA branches
  * Works with AutoGPTQ.
  * Works with text-generation-webui one-click-installers
  * Parameters: Groupsize = None. Act-order.
  * Command used to create the GPTQ:
    ```
    python llama.py ehartford_Wizard-Vicuna-30B-Uncensored c4 --wbits 4 --act-order --true-sequential --save_safetensors Wizard-Vicuna-30B-Uncensored-GPTQ-4bit-128g.act-order.safetensors
    ```

<!-- footer start -->
## Discord

For further support, and discussions on these models and AI in general, join us at:

[TheBloke AI's Discord server](https://discord.gg/Jq4vkcDakD)

## Thanks, and how to contribute.

Thanks to the [chirper.ai](https://chirper.ai) team!

I've had a lot of people ask if they can contribute. I enjoy providing models and helping people, and would love to be able to spend even more time doing it, as well as expanding into new projects like fine tuning/training.

If you're able and willing to contribute it will be most gratefully received and will help me to keep providing more models, and to start work on new AI projects.

Donaters will get priority support on any and all AI/LLM/model questions and requests, access to a private Discord room, plus other benefits.

* Patreon: https://patreon.com/TheBlokeAI
* Ko-Fi: https://ko-fi.com/TheBlokeAI

**Patreon special mentions**: Aemon Algiz, Dmitriy Samsonov, Nathan LeClaire, Trenton Dambrowitz, Mano Prime, David Flickinger, vamX, Nikolai Manek, senxiiz, Khalefa Al-Ahmad, Illia Dulskyi, Jonathan Leane, Talal Aujan, V. Lukas, Joseph William Delisle, Pyrater, Oscar Rangel, Lone Striker, Luke Pendergrass, Eugene Pentland, Sebastain Graf, Johann-Peter Hartman.

Thank you to all my generous patrons and donaters!
<!-- footer end -->

# Original model card

This is [wizard-vicuna-13b](https://huggingface.co/junelee/wizard-vicuna-13b) trained with a subset of the dataset - responses that contained alignment / moralizing were removed. The intent is to train a WizardLM that doesn't have alignment built-in, so that alignment (of any sort) can be added separately with for example with a RLHF LoRA.

Shout out to the open source AI/ML community, and everyone who helped me out.

Note:

An uncensored model has no guardrails.

You are responsible for anything you do with the model, just as you are responsible for anything you do with any dangerous object such as a knife, gun, lighter, or car.

Publishing anything this model generates is the same as publishing it yourself.

You are responsible for the content you publish, and you cannot blame the model any more than you can blame the knife, gun, lighter, or car for what you do with it.
