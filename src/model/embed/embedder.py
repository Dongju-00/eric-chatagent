import torch
import torch.nn.functional as F

from model.train.model import KoreanGPT, block_size, device
from model.train.sp_tokenizer import load_sp

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

class KoreanGPTEmbedder:
    def __init__(
        self,
        model_path=None,
        sp_prefix=None,
    ):
        if sp_prefix is None:
            sp_prefix = BASE_DIR / "data" / "sp_korean"

        if model_path is None:
            model_path = BASE_DIR / "checkpoints" / "KoreanGPT.pt"

        self.sp = load_sp(sp_prefix)

        self.model = KoreanGPT(self.sp.get_piece_size()).to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()

    @torch.no_grad()
    def embed_text(self, text: str) -> list[float]:
        ids = self.sp.encode(text, out_type=int)

        if len(ids) > block_size:
            ids = ids[-block_size:]

        idx = torch.tensor([ids], dtype=torch.long, device=device)
        T = idx.shape[1]

        x = self.model.tok_emb(idx)
        x = x + self.model.pos_emb(torch.arange(T, device=device))
        x = self.model.blocks(x)
        x = self.model.ln_f(x)

        embedding = x.mean(dim=1)
        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding[0].cpu().tolist()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(text) for text in texts]