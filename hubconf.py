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
