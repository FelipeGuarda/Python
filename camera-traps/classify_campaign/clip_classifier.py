"""
Zero-shot CLIP classifier for Chilean camera-trap species.

Builds one text embedding per species at startup (using English descriptions,
since CLIP is English-trained), then classifies image crops via cosine similarity.
"""

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

PROMPT = "a wildlife camera trap photo of a {english}"


class CLIPZeroShotClassifier:

    def __init__(self, model_name: str, species_list: list[dict]):
        """
        Args:
            model_name:   HuggingFace model ID, e.g. 'openai/clip-vit-base-patch32'
            species_list: list of dicts with keys: spanish, latin, english
        """
        self.device  = self._pick_device()
        self.species = species_list

        print(f"  Loading CLIP ({model_name}) on {self.device} …")
        self.model     = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name, use_fast=True)
        self.model.eval()

        self.text_embeddings = self._encode_species()
        print(f"  Built embeddings for {len(species_list)} species.")

    # ── Public API ────────────────────────────────────────────────────────────

    def classify(self, image: Image.Image) -> tuple[dict, float]:
        """
        Returns (species_dict, score).
            species_dict  keys: spanish, latin, english
            score         cosine similarity in [0, 1]
        """
        img_emb = self._encode_image(image)
        sims    = (img_emb @ self.text_embeddings.T).squeeze(0)
        best_i  = sims.argmax().item()
        return self.species[best_i], round(sims[best_i].item(), 4)

    @staticmethod
    def _pick_device() -> str:
        if not torch.cuda.is_available():
            return "cpu"
        try:
            # Probe: actually allocate a tensor on the GPU.
            # If the architecture is unsupported this raises a RuntimeError.
            _ = torch.zeros(1, device="cuda")
            return "cuda"
        except RuntimeError:
            major, minor = torch.cuda.get_device_capability()
            print(
                f"  [info] GPU (sm_{major}{minor}) not supported by this PyTorch build"
                f" — falling back to CPU.\n"
                f"  Install PyTorch with CUDA 12.8 support for Blackwell GPUs:\n"
                f"    pip install torch torchvision"
                f" --index-url https://download.pytorch.org/whl/cu128"
            )
            return "cpu"

    # ── Internals ─────────────────────────────────────────────────────────────

    def _encode_species(self) -> torch.Tensor:
        prompts = [PROMPT.format(english=s["english"]) for s in self.species]
        inputs  = self.processor(text=prompts, return_tensors="pt",
                                 padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            out   = self.model.text_model(**inputs)
            feats = self.model.text_projection(out.pooler_output)
        return F.normalize(feats, dim=-1)

    def _encode_image(self, image: Image.Image) -> torch.Tensor:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out   = self.model.vision_model(**inputs)
            feats = self.model.visual_projection(out.pooler_output)
        return F.normalize(feats, dim=-1)
