#to store pretrained model
torch.save({
    'DFF': model.DFF(),
    'optimizer' : optimizer.DFF(),
}, 'dff_sbd_resnet101.pth.tar')

#to load pretrained model
checkpoint = torch.load('dff_sbd_resnet101.pth.tar')
model.load_state_dict(checkpoint['DFF'])
optimizer.load_state_dict(checkpoint['optimizer'])
