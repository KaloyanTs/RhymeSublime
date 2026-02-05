import torch
from typing import Optional

from stress_model import BiLSTMStress, make_vowel_mask


class _StressPredictor:
	def __init__(self, ckpt_path: str = "stress_model.pt", device: Optional[torch.device] = None):
		self.ckpt_path = ckpt_path
		self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.model: Optional[BiLSTMStress] = None
		self.stoi = None
		self._loaded = False

	def _ensure_loaded(self):
		if self._loaded:
			return
		ckpt = torch.load(self.ckpt_path, map_location="cpu")
		self.stoi = ckpt["stoi"]
		cfg = ckpt["config"]
		self.model = BiLSTMStress(vocab_size=len(self.stoi), **cfg).to(self.device)
		self.model.load_state_dict(ckpt["state_dict"])
		self.model.eval()
		self._loaded = True

	@torch.no_grad()
	def predict(self, word: str) -> int:

		self._ensure_loaded()
		unk = self.stoi.get("<unk>", 1)
		ids = [self.stoi.get(ch, unk) for ch in word]

		x = torch.tensor([ids], dtype=torch.long, device=self.device)
		lengths = torch.tensor([len(ids)], dtype=torch.long, device=self.device)

		logits = self.model(x, lengths)  # (1, L)
		mask = make_vowel_mask([word], logits.size(1), device=self.device)
		masked_logits = logits.masked_fill(~mask, -1e9)
		idx = int(masked_logits.argmax(dim=1).item())
		return idx


_DEFAULT_PREDICTOR = _StressPredictor()


def predict(word: str, ckpt_path: Optional[str] = None) -> int:
    # print("Predicting stress for word:", word)
    global _DEFAULT_PREDICTOR
    if ckpt_path and ckpt_path != _DEFAULT_PREDICTOR.ckpt_path:
        _DEFAULT_PREDICTOR = _StressPredictor(ckpt_path=ckpt_path)
    res = _DEFAULT_PREDICTOR.predict(word)
    predicted_with_stress = word[:res] + "'" + word[res:] if res < len(word) else word + "'"
    # print("Predicted stress position:", res, "->", predicted_with_stress)
    return _DEFAULT_PREDICTOR.predict(word)

