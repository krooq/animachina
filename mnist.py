transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

dataiter = iter(trainloader)
images, labels = dataiter.next()



def torch_to_cv2(image):
    return cv2.cvtColor(np.transpose(image.cpu().numpy(), (1,2,0)), cv2.COLOR_BGR2RGB)


# show images
cv2.imshow("data", cv2.resize(torch_to_cv2(images[0]),(64,64)))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
cv2.waitKey(0)
