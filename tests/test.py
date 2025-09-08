import os
os.environ["BLT_FORCE_CPU"] = "1"
os.environ["BLT_DISABLE_MPS"] = "1"

from bytelatent.transformer import LMTransformer
from bytelatent.model.blt import ByteLatentTransformer
from bytelatent.hf import BltTokenizerAndPatcher

import json
from bytelatent.transformer import LMTransformer
from bytelatent.model.blt import ByteLatentTransformer
from bytelatent.generate_blt import generate_nocache
# entropy model args
with open("hf-weights/entropy_model/params.json") as f:
    em_cfg = json.load(f)
entropy_args = em_cfg["entropy_model"]
entropy_args["attn_impl"] = "sdpa"  # we removed xformers

# blt model args
with open("hf-weights/blt_1b/params.json") as f:
    blt_cfg = json.load(f)
blt_args = blt_cfg["model"]
blt_args["attn_impl"] = "sdpa"

entropy_model = LMTransformer.from_pretrained("facebook/blt-entropy", **entropy_args)
blt_model = ByteLatentTransformer.from_pretrained("facebook/blt-1b", **blt_args)

# Force SDPA attention everywhere on the loaded model (ignore hub defaults)
try:
    blt_model.local_encoder.attn_impl = "sdpa"
    blt_model.local_decoder.attn_impl = "sdpa"
    blt_model.global_transformer.attn_impl = "sdpa"
except AttributeError:
    pass

tok_and_patcher = BltTokenizerAndPatcher.from_pretrained("facebook/blt-1b")
tokenizer = tok_and_patcher.tokenizer_args.build()

tok_and_patcher.patcher_args.realtime_patching = True
tok_and_patcher.patcher_args.entropy_model_checkpoint_dir = "hf-weights/entropy_model"
# Force CPU everywhere
tok_and_patcher.patcher_args.patching_device = "cpu"
tok_and_patcher.patcher_args.device = "cpu"
patcher = tok_and_patcher.patcher_args.build()

print('DONE LOADING, STARTING GENERATION')

prompt = "Hello, how are you?"
prompts = [prompt]
outputs = generate_nocache(
    prompts, model=blt_model, tokenizer=tokenizer, patcher=patcher
)
text_outputs = [tokenizer.decode(t) for t in outputs]
for p, t in zip(prompts, text_outputs):
    print(f'Prompt: "{p}" Completion: "{t}"')