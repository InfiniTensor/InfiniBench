from ..Llama.model import Llama3

# the DeepSeek-R1-Distill-8B model (Llama3)
class DeepSeekR1Distill8B(Llama3):
    def __init__(self, config_path):
        super().__init__(config_path)

