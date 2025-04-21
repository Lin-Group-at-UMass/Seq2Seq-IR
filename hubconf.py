# Upload IR models to PyTorch Hub
from Spectrum2Structure.src.models import Transformer, GPT, LSTM_autoregressive, GRU

# dependencies to load and use model
dependencies = ['torch']

# entrypoint
def Spectrum2Structure(pretrained=True, model='transformer', mode='selfies', **kwargs):
    """ # This docstring shows up in hub.help()
       Spectrum2Structure model
       pretrained (bool): load pretrained weights
       model (string): model to be used (transformer, GPT, LSTM, GRU)
       mode (string): mode for model to operate in (selfies, mixture, smiles)
       """
    if model == 'transformer':
        if mode == 'selfies':
            m = Transformer(hidden_dim=768, dropout=0.1, layers=6, heads=6, batch_size=256)
            checkpoint = 'PATH/TO/DOI'
        elif mode == 'mixture':
            m = Transformer(hidden_dim=768, dropout=0.1, layers=6, heads=6, batch_size=256)
            checkpoint = 'PATH/TO/DOI'
        else:
            m = Transformer(hidden_dim=768, dropout=0.1, layers=6, heads=6, batch_size=256)
            checkpoint = 'PATH/TO/DOI'
    elif model == 'gpt':
        if mode == 'selfies':
            m = GPT(hidden_dim=400, dropout=0.1, layers=4)
            checkpoint = 'PATH/TO/DOI'
        elif mode == 'mixture':
            m = GPT(hidden_dim=400, dropout=0.1, layers=4)
            checkpoint = 'PATH/TO/DOI'
        else:
            m = GPT(hidden_dim=400, dropout=0.1, layers=4)
            checkpoint = 'PATH/TO/DOI'
    elif model == 'lstm':
        if mode == 'selfies':
            m = LSTM_autoregressive(hidden_dim=400, dropout=0.1, layers=4)
            checkpoint = 'PATH/TO/DOI'
        elif mode == 'mixture':
            m = LSTM_autoregressive(hidden_dim=400, dropout=0.1, layers=4)
            checkpoint = 'PATH/TO/DOI'
        else:
            m = LSTM_autoregressive(hidden_dim=400, dropout=0.1, layers=4)
            checkpoint = 'PATH/TO/DOI'
    elif model == 'gru':
        if mode == 'selfies':
            m = GRU(hidden_dim=400, dropout=0.1, layers=4)
            checkpoint = 'PATH/TO/DOI'
        elif mode == 'mixture':
            m = GRU(hidden_dim=400, dropout=0.1, layers=4)
            checkpoint = 'PATH/TO/DOI'
        else:
            m = GRU(hidden_dim=400, dropout=0.1, layers=4)
            checkpoint = 'PATH/TO/DOI'
