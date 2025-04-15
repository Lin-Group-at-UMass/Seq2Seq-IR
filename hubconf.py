# Upload IR models to PyTorch Hub


# Example code is below
dependencies = ['torch']

# resnet18 is the name of entrypoint
def resnet18(pretrained=False, **kwargs):
    """ # This docstring shows up in hub.help()
    Resnet18 model
    pretrained (bool): kwargs, load pretrained weights into the model
    """
    # Call the model, load pretrained weights
    model = _resnet18(pretrained=pretrained, **kwargs)
    return model

# model = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)

if pretrained:
    # For checkpoint saved in local GitHub repo, e.g. <RELATIVE_PATH_TO_CHECKPOINT>=weights/save.pth
    dirname = os.path.dirname(__file__)
    checkpoint = os.path.join(dirname, <RELATIVE_PATH_TO_CHECKPOINT>)
    state_dict = torch.load(checkpoint)
    model.load_state_dict(state_dict)

    # For checkpoint saved elsewhere
    checkpoint = 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
    model.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint, progress=False))

def IR_Transformer(pretrained=True, **kwargs):
    return 
    
def IR_GPT(pretrained=True, **kwargs):
    return 
    
def IR_LSTM(pretrained=True, **kwargs):
    return 
    
def IR_GRU(pretrained=True, **kwargs):
    return 
