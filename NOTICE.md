Third-party assets used by this project
=======================================

Salamander Grand Piano
----------------------
Yamaha C5 grand piano samples by Alexander Holm.
Licensed under Creative Commons Attribution 3.0 Unported (CC-BY 3.0).
https://archive.org/details/SalamanderGrandPianoV3

The selected velocity layers are redistributed in the Git LFS production
bundle. Attribution: "Salamander Grand Piano (Yamaha C5) by Alexander Holm."
The bundled recordings are unmodified; LexiBeat resamples and mixes them only
at playback/render time. License text: https://creativecommons.org/licenses/by/3.0/

VSCO 2 Community Edition
------------------------
Orchestral samples recorded by Sam Gossner and Simon Dalzell, with sample
cutting by Elan Hickler/Soundemote. Dedicated to the public domain under CC0 1.0.
https://github.com/sgossner/VSCO-2-CE

Curated strings, marimba and glockenspiel samples are included in the Git LFS
production bundle.

The expanded tiered library can also download and index the full VSCO 2 CE and
Versilian Community Sample Library collections. VCSL is likewise dedicated to
the public domain under CC0 1.0. The checksum-locked promoted subset is included
in the Git LFS production bundle.
https://github.com/sgossner/VCSL

FreePats World Percussion
-------------------------
World-percussion recordings by Xavimart, Gonzalo, Roberto and contributors.
Dedicated to the public domain under CC0 1.0. The bank includes cajon, bongos,
shaker, tambourine, hand clap, claves, castanets, conga, maracas and darbuka.
https://github.com/freepats/world-percussion

FreePats Spanish Classical Guitar
---------------------------------
Spanish classical-guitar recordings by Roberto for the FreePats project.
Dedicated to the public domain under CC0 1.0.
https://github.com/freepats/spanish-classical-guitar

FreePats Button Accordion HN
----------------------------
Hohner button-accordion recordings by Jeff Stauffer, processed and mapped by
michael02022 for the FreePats project. Dedicated to the public domain under
CC0 1.0.
https://github.com/freepats/button-accordion-HN

Karoryfer Fashionbass
---------------------
Sampled electric bass recorded and mapped by Karoryfer Samples.
Dedicated to the public domain under CC0 1.0.
https://github.com/sfzinstruments/karoryfer.fashionbass

Stargate Sample Pack
--------------------
Public-domain/CC0 sample collection contributed by the Stargate DAW community.
https://github.com/stargatedaw/stargate-sample-pack

Models
------
Kokoro-82M (hexgrad/Kokoro-82M) — Apache-2.0.
Chatterbox Multilingual (Resemble AI, via mlx-community/chatterbox-multilingual-v3) — MIT.

The Hugging Face Space vendors the official Chatterbox Multilingual CUDA runtime
source from resemble-ai/chatterbox revision
`5de7a54aa4e5e2baadb0182dde554908b48b85c2`. The source is distributed under
the Chatterbox MIT license; model weights remain downloaded from
`ResembleAI/chatterbox` and are not stored in this repository.
https://github.com/resemble-ai/chatterbox

Experimental model weights are downloaded on demand into the LexiBeat model
cache and are not redistributed with this repository:

IndexTTS 2.5 (IndexTeam/Bilibili; MLX conversion by vanch007) — bilibili Model
Use License Agreement. Review commercial thresholds, attribution requirements,
use restrictions and prohibited high-risk scenarios before use.
https://huggingface.co/vanch007/mlx-indextts2-2.5-8bit

VoxCPM2 (OpenBMB; mlx-community/VoxCPM2-4bit) — Apache-2.0.
https://huggingface.co/mlx-community/VoxCPM2-4bit

Qwen3-TTS CustomVoice (Qwen; MLX conversion by mlx-community) — Apache-2.0.
https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit

TADA 3B multilingual (Hume AI; based on Meta Llama 3.2) — model weights use
the Llama 3.2 Community License; the MLX runtime code is MIT.
https://huggingface.co/HumeAI/mlx-tada-3b

TADA loads the matching Llama tokenizer files from the public TADA MLX
conversion `gafiatulin/tada-3b-ml-mlx` because the Hume runtime otherwise
resolves the gated Meta tokenizer repository. No alternative model weights are
loaded. The tokenizer remains subject to the Llama 3.2 Community License.
https://huggingface.co/gafiatulin/tada-3b-ml-mlx

Fish Audio S2 Pro (Fish Audio; MLX conversion by mlx-community) — Fish Audio
Research License. Commercial use requires a separate licence.
https://huggingface.co/mlx-community/fish-audio-s2-pro-8bit
