---
datasets:
- ehartford/wizard_vicuna_70k_unfiltered
inference: false
language:
- en
license: other
model_type: llama
tags:
- uncensored
---

<!-- header start -->
<!-- 200823 -->
<div style="width: auto; margin-left: auto; margin-right: auto">
<img src="https://i.imgur.com/EBdldam.jpg" alt="TheBlokeAI" style="width: 100%; min-width: 400px; display: block; margin: auto;">
</div>
<div style="display: flex; justify-content: space-between; width: 100%;">
    <div style="display: flex; flex-direction: column; align-items: flex-start;">
        <p style="margin-top: 0.5em; margin-bottom: 0em;"><a href="https://discord.gg/theblokeai">Chat & support: TheBloke's Discord server</a></p>
    </div>
    <div style="display: flex; flex-direction: column; align-items: flex-end;">
        <p style="margin-top: 0.5em; margin-bottom: 0em;"><a href="https://www.patreon.com/TheBlokeAI">Want to contribute? TheBloke's Patreon page</a></p>
    </div>
</div>
<div style="text-align:center; margin-top: 0em; margin-bottom: 0em"><p style="margin-top: 0.25em; margin-bottom: 0em;">TheBloke's LLM work is generously supported by a grant from <a href="https://a16z.com">andreessen horowitz (a16z)</a></p></div>
<hr style="margin-top: 1.0em; margin-bottom: 1.0em;">
<!-- header end -->

# Eric Hartford's Wizard-Vicuna-30B-Uncensored GPTQ

These files are GPTQ model files for [Eric Hartford's Wizard-Vicuna-30B-Uncensored](https://huggingface.co/ehartford/Wizard-Vicuna-30B-Uncensored).

Multiple GPTQ parameter permutations are provided; see Provided Files below for details of the options provided, their parameters, and the software used to create them.

These models were quantised using hardware kindly provided by [Latitude.sh](https://www.latitude.sh/accelerate).

## Repositories available

* [GPTQ models for GPU inference, with multiple quantisation parameter options.](https://huggingface.co/TheBloke/Wizard-Vicuna-30B-Uncensored-GPTQ)
* [2, 3, 4, 5, 6 and 8-bit GGML models for CPU+GPU inference](https://huggingface.co/TheBloke/Wizard-Vicuna-30B-Uncensored-GGML)
* [Unquantised fp16 model in pytorch format, for GPU inference and for further conversions](https://huggingface.co/TheBloke/Wizard-Vicuna-30B-Uncensored-fp16)

## Prompt template: Vicuna

```
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.

USER: {prompt}
ASSISTANT:
```

## Provided files

Multiple quantisation parameters are provided, to allow you to choose the best one for your hardware and requirements.

Each separate quant is in a different branch.  See below for instructions on fetching from different branches.

| Branch | Bits | Group Size | Act Order (desc_act) | File Size | ExLlama Compatible? | Made With | Description |
| ------ | ---- | ---------- | -------------------- | --------- | ------------------- | --------- | ----------- |
| main | 4 | None | True | 16.94 GB | True | GPTQ-for-LLaMa | Most compatible option. Good inference speed in AutoGPTQ and GPTQ-for-LLaMa. Lower inference quality than other options. |
| gptq-4bit-32g-actorder_True | 4 | 32 | True | 19.44 GB | True | AutoGPTQ | 4-bit, with Act Order and group size. 32g gives highest possible inference quality, with maximum VRAM usage. Poor AutoGPTQ CUDA speed. |
| gptq-4bit-64g-actorder_True | 4 | 64 | True | 18.18 GB | True | AutoGPTQ | 4-bit, with Act Order and group size. 64g uses less VRAM than 32g, but with slightly lower accuracy. Poor AutoGPTQ CUDA speed. |
| gptq-4bit-128g-actorder_True | 4 | 128 | True | 17.55 GB | True | AutoGPTQ | 4-bit, with Act Order and group size. 128g uses even less VRAM, but with slightly lower accuracy. Poor AutoGPTQ CUDA speed. |
| gptq-8bit--1g-actorder_True | 8 | None | True | 32.99 GB | False | AutoGPTQ | 8-bit, with Act Order. No group size, to lower VRAM requirements and to improve AutoGPTQ speed. |
| gptq-8bit-128g-actorder_False | 8 | 128 | False | 33.73 GB | False | AutoGPTQ | 8-bit, with group size 128g for higher inference quality and without Act Order to improve AutoGPTQ speed. |
| gptq-3bit--1g-actorder_True | 3 | None | True | 12.92 GB | False | AutoGPTQ | 3-bit, with Act Order and no group size. Lowest possible VRAM requirements. May be lower quality than 3-bit 128g. |
| gptq-3bit-128g-actorder_False | 3 | 128 | False | 13.51 GB | False | AutoGPTQ | 3-bit, with group size 128g but no act-order. Slightly higher VRAM requirements than 3-bit None. |

## How to download from branches

- In text-generation-webui, you can add `:branch` to the end of the download name, eg `TheBloke/Wizard-Vicuna-30B-Uncensored-GPTQ:gptq-4bit-32g-actorder_True`
- With Git, you can clone a branch with:
```
git clone --branch gptq-4bit-32g-actorder_True https://huggingface.co/TheBloke/Wizard-Vicuna-30B-Uncensored-GPTQ`
```
- In Python Transformers code, the branch is the `revision` parameter; see below.

## How to easily download and use this model in [text-generation-webui](https://github.com/oobabooga/text-generation-webui).

Please make sure you're using the latest version of [text-generation-webui](https://github.com/oobabooga/text-generation-webui).

It is strongly recommended to use the text-generation-webui one-click-installers unless you know how to make a manual install.

1. Click the **Model tab**.
2. Under **Download custom model or LoRA**, enter `TheBloke/Wizard-Vicuna-30B-Uncensored-GPTQ`.
  - To download from a specific branch, enter for example `TheBloke/Wizard-Vicuna-30B-Uncensored-GPTQ:gptq-4bit-32g-actorder_True`
  - see Provided Files above for the list of branches for each option.
3. Click **Download**.
4. The model will start downloading. Once it's finished it will say "Done"
5. In the top left, click the refresh icon next to **Model**.
6. In the **Model** dropdown, choose the model you just downloaded: `Wizard-Vicuna-30B-Uncensored-GPTQ`
7. The model will automatically load, and is now ready for use!
8. If you want any custom settings, set them and then click **Save settings for this model** followed by **Reload the Model** in the top right.
  * Note that you do not need to set GPTQ parameters any more. These are set automatically from the file `quantize_config.json`.
9. Once you're ready, click the **Text Generation tab** and enter a prompt to get started!

## How to use this GPTQ model from Python code

First make sure you have [AutoGPTQ](https://github.com/PanQiWei/AutoGPTQ) installed:

`GITHUB_ACTIONS=true pip install auto-gptq`

Then try the following example code:

```python
from transformers import AutoTokenizer, pipeline, logging
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

model_name_or_path = "TheBloke/Wizard-Vicuna-30B-Uncensored-GPTQ"
model_basename = "Wizard-Vicuna-30B-Uncensored-GPTQ-4bit--1g.act.order"

use_triton = False

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

model = AutoGPTQForCausalLM.from_quantized(model_name_or_path,
        model_basename=model_basename
        use_safetensors=True,
        trust_remote_code=False,
        device="cuda:0",
        use_triton=use_triton,
        quantize_config=None)

"""
To download from a specific branch, use the revision parameter, as in this example:

model = AutoGPTQForCausalLM.from_quantized(model_name_or_path,
        revision="gptq-4bit-32g-actorder_True",
        model_basename=model_basename,
        use_safetensors=True,
        trust_remote_code=False,
        device="cuda:0",
        quantize_config=None)
"""

prompt = "Tell me about AI"
prompt_template=f'''A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.

USER: {prompt}
ASSISTANT:
'''

print("\n\n*** Generate:")

input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
output = model.generate(inputs=input_ids, temperature=0.7, max_new_tokens=512)
print(tokenizer.decode(output[0]))

# Inference can also be done using transformers' pipeline

# Prevent printing spurious transformers error when using pipeline with AutoGPTQ
logging.set_verbosity(logging.CRITICAL)

print("*** Pipeline:")
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.95,
    repetition_penalty=1.15
)

print(pipe(prompt_template)[0]['generated_text'])
```

## Compatibility

The files provided will work with AutoGPTQ (CUDA and Triton modes), GPTQ-for-LLaMa (only CUDA has been tested), and Occ4m's GPTQ-for-LLaMa fork.

ExLlama works with Llama models in 4-bit. Please see the Provided Files table above for per-file compatibility.

<!-- footer start -->
<!-- 200823 -->
## Discord

For further support, and discussions on these models and AI in general, join us at:

[TheBloke AI's Discord server](https://discord.gg/theblokeai)

## Thanks, and how to contribute.

Thanks to the [chirper.ai](https://chirper.ai) team!

I've had a lot of people ask if they can contribute. I enjoy providing models and helping people, and would love to be able to spend even more time doing it, as well as expanding into new projects like fine tuning/training.

If you're able and willing to contribute it will be most gratefully received and will help me to keep providing more models, and to start work on new AI projects.

Donaters will get priority support on any and all AI/LLM/model questions and requests, access to a private Discord room, plus other benefits.

* Patreon: https://patreon.com/TheBlokeAI
* Ko-Fi: https://ko-fi.com/TheBlokeAI

**Special thanks to**: Aemon Algiz.

**Patreon special mentions**: Sam, theTransient, Jonathan Leane, Steven Wood, webtim, Johann-Peter Hartmann, Geoffrey Montalvo, Gabriel Tamborski, Willem Michiel, John Villwock, Derek Yates, Mesiah Bishop, Eugene Pentland, Pieter, Chadd, Stephen Murray, Daniel P. Andersen, terasurfer, Brandon Frisco, Thomas Belote, Sid, Nathan LeClaire, Magnesian, Alps Aficionado, Stanislav Ovsiannikov, Alex, Joseph William Delisle, Nikolai Manek, Michael Davis, Junyu Yang, K, J, Spencer Kim, Stefan Sabev, Olusegun Samson, transmissions 11, Michael Levine, Cory Kujawski, Rainer Wilmers, zynix, Kalila, Luke @flexchar, Ajan Kanaga, Mandus, vamX, Ai Maven, Mano Prime, Matthew Berman, subjectnull, Vitor Caleffi, Clay Pascal, biorpg, alfie_i, 阿明, Jeffrey Morgan, ya boyyy, Raymond Fosdick, knownsqashed, Olakabola, Leonard Tan, ReadyPlayerEmma, Enrico Ros, Dave, Talal Aujan, Illia Dulskyi, Sean Connelly, senxiiz, Artur Olbinski, Elle, Raven Klaugh, Fen Risland, Deep Realms, Imad Khwaja, Fred von Graf, Will Dee, usrbinkat, SuperWojo, Alexandros Triantafyllidis, Swaroop Kallakuri, Dan Guido, John Detwiler, Pedro Madruga, Iucharbius, Viktor Bowallius, Asp the Wyvern, Edmond Seymore, Trenton Dambrowitz, Space Cruiser, Spiking Neurons AB, Pyrater, LangChain4j, Tony Hughes, Kacper Wikieł, Rishabh Srivastava, David Ziegler, Luke Pendergrass, Andrey, Gabriel Puliatti, Lone Striker, Sebastain Graf, Pierre Kircher, Randy H, NimbleBox.ai, Vadim, danny, Deo Leter


Thank you to all my generous patrons and donaters!

And thank you again to a16z for their generous grant.

<!-- footer end -->

# Original model card: Eric Hartford's Wizard-Vicuna-30B-Uncensored


This is [wizard-vicuna-13b](https://huggingface.co/junelee/wizard-vicuna-13b) trained with a subset of the dataset - responses that contained alignment / moralizing were removed. The intent is to train a WizardLM that doesn't have alignment built-in, so that alignment (of any sort) can be added separately with for example with a RLHF LoRA.

Shout out to the open source AI/ML community, and everyone who helped me out.

Note:

An uncensored model has no guardrails.

You are responsible for anything you do with the model, just as you are responsible for anything you do with any dangerous object such as a knife, gun, lighter, or car.

Publishing anything this model generates is the same as publishing it yourself.

You are responsible for the content you publish, and you cannot blame the model any more than you can blame the knife, gun, lighter, or car for what you do with it.
