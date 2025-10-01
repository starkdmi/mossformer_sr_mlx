import mlx.nn as nn
from generator import Mossformer, Generator

class AttrDict(dict):
    """A dictionary that allows attribute-style access."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

class MossFormer2_SR_48K(nn.Module):
    """
    The MossFormer2_SR_48K model for speech super-resolution.

    This class encapsulates the functionality of the MossFormer2 and HiFi-Gan
    Generator within a higher-level model. It processes input audio data to produce
    higher-resolution outputs.
    """

    def __init__(self, args):
        super().__init__()
        # Initialize the Mossformer model
        self.model_m = Mossformer()
        # Initialize the Generator with args
        self.model_g = Generator(args)

    def __call__(self, x):
        """
        Forward pass through the model.

        Arguments
        ---------
        x : mx.array
            Input tensor of dimension [B, N, S], where B is the batch size,
            N is the number of mel bins (80 in this case), and S is the
            sequence length (e.g., time frames).

        Returns
        -------
        outputs : mx.array
            Bandwidth expanded audio output tensor from the model.
        """
        x = self.model_m(x)  # Get outputs from Mossformer
        outputs = self.model_g(x)  # Generate audio with HiFi-GAN
        return outputs